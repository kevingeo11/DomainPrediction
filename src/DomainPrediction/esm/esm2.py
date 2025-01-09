import torch
import esm
from tqdm import tqdm
import numpy as np


class ESM2():
    def __init__(self, model_path, device='cpu') -> None:
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.device = device

        if 't6_8M' in model_path:
            self.rep_layer = 6
            self.emb_dim = 320
        elif 't30_150M' in model_path:
            self.rep_layer = 30
            self.emb_dim = 640
        elif 't33_650M' in model_path:
            self.rep_layer = 33
            self.emb_dim = 1280
        else:
            raise Exception('I need to work on this. Feel free to extend :)')

        if self.device == 'gpu':
            self.model.cuda()

        self.tok_to_idx = self.alphabet.tok_to_idx
        self.idx_to_tok = {v:k for k,v in self.tok_to_idx.items()}

    def get_res(self, sequence, rep_layer=None):
        if rep_layer is None:
            rep_layer = self.rep_layer

        data = [
            ("protein1", sequence)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[rep_layer], return_contacts=True)

        return results

    def get_res_batch(self, sequences, rep_layer=None):
        if rep_layer is None:
            rep_layer = self.rep_layer

        data = [
            (f"P{i+1}", seq) for i, seq in enumerate(sequences)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[rep_layer], return_contacts=True)

        return results, batch_lens

    def get_logits(self, sequence):

        results = self.get_res(sequence)
        return results['logits']

    def get_prob(self, sequence):
        logits = self.get_logits(sequence)
        prob = torch.nn.functional.softmax(logits, dim=-1)[0, 1:-1, :] # 1st and last are start and end tokens

        return prob.cpu().numpy()
    
    def get_embeddings_mean(self, sequences):
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq)
            embeddings.append(rep['representations'][self.rep_layer][:,1:-1,:].mean(1).cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
    
    def get_embeddings_flatten(self, sequences):
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq)
            embeddings.append(rep['representations'][self.rep_layer][:,1:-1,:].cpu().numpy()[0].flatten())

        embeddings = np.stack(embeddings, axis=0)

        return embeddings
    
    def __get_embeddings_full(self, sequences):
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq)
            embeddings.append(rep['representations'][self.rep_layer][:,1:-1,:].cpu().numpy()[0])

        return embeddings
    
    def get_embeddings_cls(self, sequences):
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq)
            embeddings.append(rep['representations'][self.rep_layer][:, 0, :].cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
    
    def get_embeddings_feature_pool(self, sequences, pool='mean'):
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq)
            if pool == 'mean':
                embeddings.append(rep['representations'][self.rep_layer][:,1:-1,:].mean(-1).cpu().numpy())
            elif pool == 'sum':
                embeddings.append(rep['representations'][self.rep_layer][:,1:-1,:].sum(-1).cpu().numpy())
            else:
                raise Exception('pool can only take values mean or sum')
            
        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
    
    def compute_perplexity(self, sequence, mask_token='<mask>'):
        '''
            pseudoperplexity(x) = exp( -1/L \sum_{i=1}_{L} [log( p(x_{i}|x_{j!=i}) )] )
            
        '''
        
        sum_log = 0
        for pos in range(len(sequence)):
            masked_query = list(sequence)
            assert mask_token not in masked_query
            masked_query[pos] = mask_token
            masked_query = ''.join(masked_query)
            prob = self.get_prob(sequence=masked_query)

            assert prob.shape[0] == len(sequence)

            prob_pos = np.log(prob[pos, self.tok_to_idx[sequence[pos]]])
            
            sum_log += prob_pos

        return np.exp(-1*sum_log/len(sequence))
    
    def get_log_prob(self, sequence):
        logits = self.get_logits(sequence)
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

                idx_mt = self.tok_to_idx[aa_mt]
                idx_wt = self.tok_to_idx[aa_wt]
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

                idx_mt = self.tok_to_idx[aa_mt]
                idx_wt = self.tok_to_idx[aa_wt]
                score += masked_log_prob[i, idx_mt] - masked_log_prob[i, idx_wt]


        return score, n_muts

    

class ESM2_iterate():
    def __init__(self, model_path, device='cpu') -> None:
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()
        self.device = device

        if self.device == 'gpu':
            self.model.cuda()

        self.tok_to_idx = self.alphabet.tok_to_idx
        self.idx_to_tok = {v:k for k,v in self.tok_to_idx.items()}

    def random_fill(self, sequence, infill_idx, fix=2, verbose=False):
        masked_query = ''.join(['<mask>' if i in infill_idx else sequence[i] for i in range(len(sequence))])

        data = [
            ("protein1", masked_query)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        ## Random Mask Approach
        with torch.no_grad():
            for i in range(len(infill_idx)):
                if verbose:
                    print(f'run {i}')
                
                results =self. model(batch_tokens, repr_layers=[33], return_contacts=True)
                prob = torch.nn.functional.softmax(results['logits'], dim=-1)
                masked_positions = torch.nonzero(batch_tokens == 32, as_tuple=False)
                
                if verbose:
                    print(f'num maxed pos {masked_positions.size(0)}')
                
                if masked_positions.size(0) > fix:
                    selected_positions = masked_positions[torch.randperm(masked_positions.size(0))[:fix]]
                    batch_tokens[0, selected_positions[:, 1]] = torch.multinomial(prob[:, selected_positions[:, 1], :][0], 1).flatten()
                else:
                    selected_positions = masked_positions
                    batch_tokens[0, selected_positions[:, 1]] = torch.multinomial(prob[:, selected_positions[:, 1], :][0], 1).flatten()

                if verbose:
                    print(f" T domain : {''.join([self.idx_to_tok[i] if self.idx_to_tok[i] != '<mask>' else '_' for i in batch_tokens[0, 1:-1][infill_idx].numpy()])}")

                if torch.nonzero(batch_tokens == 32, as_tuple=False).size(0) == 0:
                    break

        new_sequence = ''.join([self.idx_to_tok[i] for i in batch_tokens.cpu().numpy()[0][1:-1]])

        assert len(sequence) == len(new_sequence)
        assert ''.join([sequence[i] for i in range(len(sequence)) if i not in infill_idx]) == ''.join([new_sequence[i] for i in range(len(new_sequence)) if i not in infill_idx])

        return new_sequence
    
    def entropy_fill(self, sequence, infill_idx, fix=2, verbose=False):
        masked_query = ''.join(['<mask>' if i in infill_idx else sequence[i] for i in range(len(sequence))])

        data = [
            ("protein1", masked_query)
        ]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

        if self.device == 'gpu':
            batch_tokens = batch_tokens.cuda()

        ## Min Entropy position
        with torch.no_grad():
            for i in range(len(infill_idx)):
                if verbose:
                    print(f'run {i}')

                results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
                prob = torch.nn.functional.softmax(results['logits'], dim=-1)
                masked_positions = torch.nonzero(batch_tokens == 32, as_tuple=False)

                if verbose:
                    print(f'num maxed pos {masked_positions.size(0)}')
                
                if masked_positions.size(0) > fix:
                    masked_probs = prob[0 ,masked_positions[:, 1], :]
                    entropies = -torch.sum(masked_probs * torch.log(masked_probs + 1e-10), dim=-1)
                    indices = torch.argsort(entropies)[:fix]
                    selected_positions = masked_positions[:, 1][indices]
                    batch_tokens[0, selected_positions] = torch.multinomial(prob[:, selected_positions, :][0], 1).flatten()
                else:
                    selected_positions = masked_positions[:, 1]
                    batch_tokens[0, selected_positions] = torch.multinomial(prob[:, selected_positions, :][0], 1).flatten()
                
                if verbose:
                    print(f" T domain : {''.join([self.idx_to_tok[i] if self.idx_to_tok[i] != '<mask>' else '_' for i in batch_tokens[0, 1:-1][infill_idx].numpy()])}")

                if torch.nonzero(batch_tokens == 32, as_tuple=False).size(0) == 0:
                    break

        new_sequence = ''.join([self.idx_to_tok[i] for i in batch_tokens.cpu().numpy()[0][1:-1]])

        assert len(sequence) == len(new_sequence)
        assert ''.join([sequence[i] for i in range(len(sequence)) if i not in infill_idx]) == ''.join([new_sequence[i] for i in range(len(new_sequence)) if i not in infill_idx])

        return new_sequence