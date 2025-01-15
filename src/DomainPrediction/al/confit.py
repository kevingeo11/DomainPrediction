import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from peft import LoraConfig, get_peft_model

import numpy as np

env = os.environ['CONDA_DEFAULT_ENV']
if env == 'workspace':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
elif env == 'workspace-esm':
    import esm
else:
    raise Exception('Who are you?')


class ProteinFunDatasetContrast(Dataset):
    def __init__(self, df, wt):
        self.seq, self.y = df['seq'].to_numpy(), df['fitness_raw'].to_numpy()
        self.wt = np.array([wt]*self.seq.shape[0], dtype='object')
        self.n_mut = df['n_mut'].to_numpy()

        self.positions = []
        for _, row in df.iterrows():
            mt_sequence = row['seq']
            pos = []
            for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt)):
                if aa_wt != aa_mt:
                    ## mutation pos
                    pos.append(i)

            assert len(pos) == row['n_mut']

            self.positions.append(np.array(pos))

        assert len(self.positions) == self.seq.shape[0]
    
    def __len__(self):
        return self.seq.shape[0]
    
    def __getitem__(self, idx):
        return self.seq[idx], self.y[idx], self.wt[idx], self.positions[idx], self.n_mut[idx]
    
    @staticmethod
    def collate_fn(data):
        seq = np.array([x[0] for x in data], dtype='object')
        y = torch.tensor([x[1] for x in data])
        wt = np.array([x[2] for x in data], dtype='object')
        pos = [x[3] for x in data]
        n_mut = np.array([x[4] for x in data])
        return seq, y, wt, pos, n_mut


class ESM2ConFit(pl.LightningModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.basemodel, self.alphabet = esm.pretrained.load_model_and_alphabet(config['model_path'])
        self.model_reg, _ = esm.pretrained.load_model_and_alphabet(config['model_path'])
        self.batch_converter = self.alphabet.get_batch_converter()
        
        for pm in self.model_reg.parameters():
            pm.requires_grad = False
        self.model_reg.eval()
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias='all'
        )
        
        self.model = get_peft_model(self.basemodel, peft_config)
        
        if config['device'] == 'gpu':
            self.model.cuda()
            self.model_reg.cuda()

        self.lambda_reg = config['lambda']

        self.accumulate_batch_loss_train = []
        self.accumulate_batch_loss_val = []
        self.debug=True

    def forward(self, batch, batch_tokens_masked, batch_tokens, batch_tokens_wt):
        mt_seq, _, wt_seq, pos, n_mut = batch
        
        logits = self.model(batch_tokens_masked)['logits']
        log_probs = torch.log_softmax(logits, dim=-1)

        scores = torch.zeros(log_probs.shape[0])
        if self.config['device'] == 'gpu':
            scores = scores.cuda()

        for i in range(log_probs.shape[0]):
            scores[i] = torch.sum(log_probs[i, pos[i]+1, batch_tokens[i][pos[i]+1]] - log_probs[i, pos[i]+1, batch_tokens_wt[i][pos[i]+1]])
        
        return scores, logits
    
    def BT_loss(self, scores, y):
        loss = torch.tensor(0.)
        if self.config['device'] == 'gpu':
            loss = loss.cuda()

        for i in range(len(scores)):
            for j in range(i, len(scores)):
                if y[i] > y[j]:
                    loss += torch.log(1 + torch.exp(scores[j]-scores[i]))
                else:
                    loss += torch.log(1 + torch.exp(scores[i]-scores[j]))
        return loss

    def training_step(self, batch, batch_idx):
        mt_seq, y, wt_seq, pos, n_mut = batch
        data = [
            (f'P{i}', wt_i) for i, wt_i in enumerate(wt_seq)
            ]
        _, _, batch_tokens_wt = self.batch_converter(data)

        data = [
            (f'P{i}', s) for i, s in enumerate(mt_seq)
            ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        batch_tokens_masked = batch_tokens.clone()
        for i in range(batch_tokens.shape[0]):
            if len(pos[i]) > 0:
                batch_tokens_masked[i, pos[i]+1] = self.alphabet.mask_idx
        
        if self.config['device'] == 'gpu':
            batch_tokens_masked = batch_tokens_masked.cuda()

        y_hat, logits = self(batch, batch_tokens_masked, batch_tokens, batch_tokens_wt)

        bt_loss = self.BT_loss(y_hat, y)

        if self.config['device'] == 'gpu':
            batch_tokens_wt = batch_tokens_wt.cuda()

        logits_reg = self.model_reg(batch_tokens_wt)['logits']

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = bt_loss + self.lambda_reg*l_reg

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_train.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        mt_seq, y, wt_seq, pos, n_mut = batch
        data = [
            (f'P{i}', wt_i) for i, wt_i in enumerate(wt_seq)
            ]
        _, _, batch_tokens_wt = self.batch_converter(data)

        data = [
            (f'P{i}', s) for i, s in enumerate(mt_seq)
            ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        batch_tokens_masked = batch_tokens.clone()
        for i in range(batch_tokens.shape[0]):
            if len(pos[i]) > 0:
                batch_tokens_masked[i, pos[i]+1] = self.alphabet.mask_idx
        
        if self.config['device'] == 'gpu':
            batch_tokens_masked = batch_tokens_masked.cuda()

        y_hat, logits = self(batch, batch_tokens_masked, batch_tokens, batch_tokens_wt)

        bt_loss = self.BT_loss(y_hat, y)

        if self.config['device'] == 'gpu':
            batch_tokens_wt = batch_tokens_wt.cuda()

        logits_reg = self.model_reg(batch_tokens_wt)['logits']

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = bt_loss + self.lambda_reg*l_reg

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_val.append(loss.item())

    def trainmodel(self, df, wt, val=None, debug=True):
        self.model.train()
        
        self.debug = debug

        train_dataset = ProteinFunDatasetContrast(df, wt)

        val_loader = None
        if val is not None:
            val_dataset = ProteinFunDatasetContrast(val, wt)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=True)
        # train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=False)

        callbacks = None
        if self.config['early_stopping']:
            callbacks = []
            earlystopping_callback = EarlyStopping(monitor="val/loss", patience=self.config['patience'], verbose=False, mode="min")
            callbacks.append(earlystopping_callback)


        trainer = pl.Trainer(max_epochs=self.config['epoch'], callbacks=callbacks,
                                accelerator="auto",
                                enable_progress_bar=False,
                                enable_model_summary=True,
                                precision="16-mixed",
                                accumulate_grad_batches=self.config['accumulate_batch_size']
                                )
        
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def sanity_check(self, df, wt):
        dataset = ProteinFunDatasetContrast(df, wt)
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=False)

        y_pred_1 = []
        for batch in loader:
            mt_seq, y, wt_seq, pos, n_mut = batch
            data = [
                (f'P{i}', wt_i) for i, wt_i in enumerate(wt_seq)
                ]
            _, _, batch_tokens_wt = self.batch_converter(data)

            data = [
                (f'P{i}', s) for i, s in enumerate(mt_seq)
                ]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

            batch_tokens_masked = batch_tokens.clone()
            for i in range(batch_tokens.shape[0]):
                if len(pos[i]) > 0:
                    batch_tokens_masked[i, pos[i]+1] = self.alphabet.mask_idx
            
            if self.config['device'] == 'gpu':
                batch_tokens_masked = batch_tokens_masked.cuda()

            with torch.no_grad():
                y_hat, _ = self(batch, batch_tokens_masked, batch_tokens, batch_tokens_wt)

            y_pred_1.append(y_hat.cpu().numpy())

        y_pred_1 = np.concatenate(y_pred_1)

        y_pred_2 = []
        for i, row in df.iterrows():
            mt_sequence = row['seq']
            score, n_muts = self.get_masked_marginal(mt_sequence, wt)
            assert n_muts == row['n_mut']

            y_pred_2.append(score)

        y_pred_2 = np.array(y_pred_2)

        np.allclose(y_pred_1, y_pred_2, atol=1e-3)
            
    def on_train_epoch_start(self):
        self.accumulate_batch_loss_train.clear()
        self.accumulate_batch_loss_val.clear()
    
    def on_train_epoch_end(self):
        if self.current_epoch % self.config['print_every_n_epoch'] == 0 and self.debug:
            print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} val loss: {np.mean(self.accumulate_batch_loss_val)}')

    def on_train_end(self):
        print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} val loss: {np.mean(self.accumulate_batch_loss_val)}')

    def get_log_prob(self, sequence):
        data = [
            ("protein1", sequence)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.config['device'] == 'gpu':
            batch_tokens = batch_tokens.cuda()
            self.model = self.model.cuda()

        with torch.no_grad():
            logits = self.model(batch_tokens)['logits']

        log_prob = torch.log_softmax(logits, dim=-1)[0,1:-1,:]

        return log_prob.cpu().numpy()
    
    def get_wildtype_marginal(self, mt_sequence, wt_sequence, wt_log_prob=None):
        if wt_log_prob is None:
            assert len(wt_sequence) == len(mt_sequence)
            wt_log_prob = self.get_log_prob(sequence=wt_sequence)

        assert wt_log_prob.shape[0] == len(wt_sequence) == len(mt_sequence)

        n_muts = 0
        score = 0
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos
                n_muts += 1

                idx_mt = self.alphabet.get_idx(aa_mt)
                idx_wt = self.alphabet.get_idx(aa_wt)
                score += wt_log_prob[i, idx_mt] - wt_log_prob[i, idx_wt]


        return score, n_muts
    
    def get_masked_marginal(self, mt_sequence, wt_sequence, mask_token = '<mask>'):

        assert len(wt_sequence) == len(mt_sequence)

        n_muts = 0
        mask_positions = []
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos
                n_muts += 1
                mask_positions.append(i)

        assert len(mask_positions) == n_muts
        masked_query = list(wt_sequence)
        for _pos in mask_positions:
            masked_query[_pos] = mask_token
        masked_sequence = ''.join(masked_query)

        masked_log_prob = self.get_log_prob(sequence=masked_sequence)
        
        score = 0
        _idx = 0
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos

                assert mask_positions[_idx] == i
                _idx += 1

                idx_mt = self.alphabet.get_idx(aa_mt)
                idx_wt = self.alphabet.get_idx(aa_wt)
                score += masked_log_prob[i, idx_mt] - masked_log_prob[i, idx_wt]


        return score, n_muts
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
    
    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )



class ESMCConFit(pl.LightningModule):
    def __init__(self, name, config) -> None:
        super().__init__()
        self.config = config

        if name == 'esmc_300m':
            self.basemodel = ESMC.from_pretrained(name)
            self.model_reg = ESMC.from_pretrained(name)
            self.emb_dim = 960
        elif name == 'esmc_600m':
            self.basemodel = ESMC.from_pretrained(name)
            self.model_reg = ESMC.from_pretrained(name)
            self.emb_dim = 1152
        else:
            raise Exception('Check ESMC name')
        
        for pm in self.model_reg.parameters():
            pm.requires_grad = False
        self.model_reg.eval()
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=["out_proj"],
        )
        
        self.model = get_peft_model(self.basemodel, peft_config)

        for name, pm in self.model.named_parameters():
            if 'q_ln' in name or 'k_ln' in name:
                pm.requires_grad = True
        
        if config['device'] == 'gpu':
            self.model.cuda()
            self.model_reg.cuda()

        self.lambda_reg = config['lambda']

        self.accumulate_batch_loss_train = []
        self.accumulate_batch_loss_val = []
        self.debug=True

    def forward(self, batch, batch_tokens_masked, batch_tokens, batch_tokens_wt):
        mt_seq, _, wt_seq, pos, n_mut = batch
        
        output = self.model(batch_tokens_masked)
        logits = output.sequence_logits
        log_probs = torch.log_softmax(logits, dim=-1)

        scores = torch.zeros(log_probs.shape[0])
        if self.config['device'] == 'gpu':
            scores = scores.cuda()

        for i in range(log_probs.shape[0]):
            scores[i] = torch.sum(log_probs[i, pos[i]+1, batch_tokens[i][pos[i]+1]] - log_probs[i, pos[i]+1, batch_tokens_wt[i][pos[i]+1]])
        
        return scores, logits
    
    def BT_loss(self, scores, y):
        loss = torch.tensor(0.)
        if self.config['device'] == 'gpu':
            loss = loss.cuda()

        for i in range(len(scores)):
            for j in range(i, len(scores)):
                if y[i] > y[j]:
                    loss += torch.log(1 + torch.exp(scores[j]-scores[i]))
                else:
                    loss += torch.log(1 + torch.exp(scores[i]-scores[j]))
        return loss

    def training_step(self, batch, batch_idx):
        mt_seq, y, wt_seq, pos, n_mut = batch
        batch_tokens_wt = self.model._tokenize(wt_seq)
        batch_tokens = self.model._tokenize(mt_seq)

        batch_tokens_masked = batch_tokens.clone()
        for i in range(batch_tokens.shape[0]):
            if len(pos[i]) > 0:
                batch_tokens_masked[i, pos[i]+1] = self.model.tokenizer.mask_token_id
        
        if self.config['device'] == 'gpu':
            batch_tokens_masked = batch_tokens_masked.cuda()

        y_hat, logits = self(batch, batch_tokens_masked, batch_tokens, batch_tokens_wt)

        bt_loss = self.BT_loss(y_hat, y)

        if self.config['device'] == 'gpu':
            batch_tokens_wt = batch_tokens_wt.cuda()

        output = self.model_reg(batch_tokens_wt)
        logits_reg = output.sequence_logits

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = bt_loss + self.lambda_reg*l_reg

        # print(f'contrast loss: {bt_loss.item()} | reg loss: {l_reg.item()} | loss: {loss.item()}')

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_train.append(loss.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        mt_seq, y, wt_seq, pos, n_mut = batch
        batch_tokens_wt = self.model._tokenize(wt_seq)
        batch_tokens = self.model._tokenize(mt_seq)

        batch_tokens_masked = batch_tokens.clone()
        for i in range(batch_tokens.shape[0]):
            if len(pos[i]) > 0:
                batch_tokens_masked[i, pos[i]+1] = self.model.tokenizer.mask_token_id
        
        if self.config['device'] == 'gpu':
            batch_tokens_masked = batch_tokens_masked.cuda()

        y_hat, logits = self(batch, batch_tokens_masked, batch_tokens, batch_tokens_wt)

        bt_loss = self.BT_loss(y_hat, y)

        if self.config['device'] == 'gpu':
            batch_tokens_wt = batch_tokens_wt.cuda()

        output = self.model_reg(batch_tokens_wt)
        logits_reg = output.sequence_logits

        creterion_reg = torch.nn.KLDivLoss(reduction='batchmean')
        probs = torch.softmax(logits, dim=-1)
        probs_reg = torch.softmax(logits_reg, dim=-1)
        l_reg = creterion_reg(probs_reg.log().cuda(), probs)

        loss = bt_loss + self.lambda_reg*l_reg

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=y.shape[0])
        self.accumulate_batch_loss_val.append(loss.item())

    def trainmodel(self, df, wt, val=None, debug=True):
        self.model.train()
        
        self.debug = debug

        train_dataset = ProteinFunDatasetContrast(df, wt)

        val_loader = None
        if val is not None:
            val_dataset = ProteinFunDatasetContrast(val, wt)
            val_loader = DataLoader(val_dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=False)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=True)
        # train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=False)

        callbacks = None
        if self.config['early_stopping']:
            callbacks = []
            earlystopping_callback = EarlyStopping(monitor="val/loss", patience=self.config['patience'], verbose=False, mode="min")
            callbacks.append(earlystopping_callback)


        trainer = pl.Trainer(max_epochs=self.config['epoch'], callbacks=callbacks,
                                accelerator="auto",
                                enable_progress_bar=False,
                                enable_model_summary=True,
                                precision="bf16-mixed",
                                accumulate_grad_batches=self.config['accumulate_batch_size']
                                )
        
        trainer.fit(model=self, train_dataloaders=train_loader, val_dataloaders=val_loader)

    def sanity_check(self, df, wt):
        dataset = ProteinFunDatasetContrast(df, wt)
        loader = DataLoader(dataset, batch_size=self.config['batch_size'], collate_fn=ProteinFunDatasetContrast.collate_fn, shuffle=False)

        y_pred_1 = []
        for batch in loader:
            mt_seq, y, wt_seq, pos, n_mut = batch
            batch_tokens_wt = self.model._tokenize(wt_seq)
            batch_tokens = self.model._tokenize(mt_seq)

            batch_tokens_masked = batch_tokens.clone()
            for i in range(batch_tokens.shape[0]):
                if len(pos[i]) > 0:
                    batch_tokens_masked[i, pos[i]+1] = self.model.tokenizer.mask_token_id
            
            if self.config['device'] == 'gpu':
                batch_tokens_masked = batch_tokens_masked.cuda()

            with torch.no_grad():
                y_hat, _ = self(batch, batch_tokens_masked, batch_tokens, batch_tokens_wt)

            y_pred_1.append(y_hat.cpu().numpy())

        y_pred_1 = np.concatenate(y_pred_1)

        y_pred_2 = []
        for i, row in df.iterrows():
            mt_sequence = row['seq']
            score, n_muts = self.get_masked_marginal(mt_sequence, wt)
            assert n_muts == row['n_mut']

            y_pred_2.append(score)

        y_pred_2 = np.array(y_pred_2)

        np.allclose(y_pred_1, y_pred_2, atol=1e-3)
            
    def on_train_epoch_start(self):
        self.accumulate_batch_loss_train.clear()
        self.accumulate_batch_loss_val.clear()
    
    def on_train_epoch_end(self):
        if self.current_epoch % self.config['print_every_n_epoch'] == 0 and self.debug:
            print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} val loss: {np.mean(self.accumulate_batch_loss_val)}')

    def on_train_end(self):
        print(f'Epoch: {self.current_epoch}: train loss: {np.mean(self.accumulate_batch_loss_train)} val loss: {np.mean(self.accumulate_batch_loss_val)}')

    def get_log_prob(self, sequence):
        esm_protein = ESMProtein(sequence=sequence)

        if self.config['device'] == 'gpu':
            self.model = self.model.cuda()

        esm_tensor = self.model.encode(esm_protein)

        with torch.no_grad():
            results = self.model.logits(
                esm_tensor, LogitsConfig(sequence=True, return_embeddings=False)
            )

        logits = results.logits.sequence

        log_prob = torch.log_softmax(logits[0, 1:-1, :33], dim=-1)

        return log_prob.to(torch.float32).cpu().numpy()
    
    def get_wildtype_marginal(self, mt_sequence, wt_sequence, wt_log_prob=None):
        if wt_log_prob is None:
            assert len(wt_sequence) == len(mt_sequence)
            wt_log_prob = self.get_log_prob(sequence=wt_sequence)

        assert wt_log_prob.shape[0] == len(wt_sequence) == len(mt_sequence)

        n_muts = 0
        score = 0
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos
                n_muts += 1

                idx_mt = self.model.tokenizer.convert_tokens_to_ids(aa_mt)
                idx_wt = self.model.tokenizer.convert_tokens_to_ids(aa_wt)
                score += wt_log_prob[i, idx_mt] - wt_log_prob[i, idx_wt]


        return score, n_muts
    
    def get_masked_marginal(self, mt_sequence, wt_sequence, mask_token = '_'):

        assert len(wt_sequence) == len(mt_sequence)

        n_muts = 0
        mask_positions = []
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos
                n_muts += 1
                mask_positions.append(i)

        assert len(mask_positions) == n_muts
        masked_query = list(wt_sequence)
        for _pos in mask_positions:
            masked_query[_pos] = mask_token
        masked_sequence = ''.join(masked_query)

        masked_log_prob = self.get_log_prob(sequence=masked_sequence)
        
        score = 0
        _idx = 0
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos

                assert mask_positions[_idx] == i
                _idx += 1

                idx_mt = self.model.tokenizer.convert_tokens_to_ids(aa_mt)
                idx_wt = self.model.tokenizer.convert_tokens_to_ids(aa_wt)
                score += masked_log_prob[i, idx_mt] - masked_log_prob[i, idx_wt]


        return score, n_muts
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
    
    def print_trainable_parameters(self, model):
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )