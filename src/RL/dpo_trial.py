import sys
sys.path.append('..')

import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from DomainPrediction.utils import helper
from DomainPrediction.eval import metrics
from DomainPrediction.al import top_model as topmodel
from DomainPrediction.al.embeddings import one_hot_encode
from DomainPrediction.protein.base import BaseProtein
from DomainPrediction.utils.constants import *

sys.path.append('../../esm')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig, GenerationConfig, ESMProteinTensor
from esm.utils.structure.protein_chain import ProteinChain

from esm.utils.generation import (
    _stack_protein_tensors,
)

class DPOTrainer:
    def __init__(self, model, model_ref, device='cpu') -> None:
        self.device = device

        # if self.device == 'gpu':
        #     self.model = model.to("cuda:0")
        #     self.model_ref = model_ref.to("cuda:1")
        # else:
        #     self.model = model.to("cpu")
        #     self.model_ref = model_ref.to("cpu")

        self.model = model
        self.model_ref =  model_ref

        self.model_ref.eval()
        for pm in self.model_ref.parameters():
            pm.requires_grad = False

        for name, param in self.model.named_parameters():
            if name in [
                "encoder.sequence_embed.weight", 
                "output_heads.sequence_head.0.weight", 
                "output_heads.sequence_head.0.bias",
                "output_heads.sequence_head.2.weight",
                "output_heads.sequence_head.2.bias",
                "output_heads.sequence_head.3.weight",
                "output_heads.sequence_head.3.bias"
            ]:
                param.requires_grad = True
            else:
                param.requires_grad = False

        ## need to add peft here ? Do we ?
    
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

    def generate_batch(self, sequence_prompt, structure_prompt, batch_size, ref=False):
        sequence_prediction_config = GenerationConfig(
            track="sequence", 
            num_steps=sequence_prompt.count("_") // 2, 
            temperature=0.5
        )
        esm_protein = ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt)
        
        generated_proteins = []
        for _ in tqdm(range(batch_size)):
            generated_protein = self.model.generate(esm_protein, sequence_prediction_config)
            generated_proteins.append(generated_protein.sequence)

        return generated_proteins
    
    def get_mask_positions(self, sequence_prompt, mask_var='_'):
        positions = []
        for i in range(len(sequence_prompt)):
            if sequence_prompt[i] == mask_var:
                positions.append(i)

        return np.array(positions)
    
    def __get_n_mutations(self, sequence):
        base = 'APSEDAYPRATYEAPEGETEQLLAGIWMDLLQVDRVGRHDSFFELGGHSLLAVRLLGRLRQHGLGLQMRDLFEAPVLAELATRLRPYQPLEVPANGITPDTTVLTPEMLPLVTLS'
        assert len(base) == len(sequence)

        count = 0
        for aa1, aa2 in zip(base, sequence):
            if aa1 != aa2:
                count += 1

        return -1*count
    
    def get_property_batch(self, batch):
        for item in batch:
            item['property'] = self.__get_n_mutations(item['masked_sequence'])

        return batch
    
    def get_PLL(self, item, ref=False):
        '''
            We always feed in the sequence prompt
            We get Log Prob(A_C) -  entire T region is masked
        '''
        proteins = [ESMProtein(sequence=item['sequence_prompt'], 
                        coordinates=item['structure_prompt'])]
        
        if ref:
            input_tokens = [self.model_ref.encode(protein) for protein in proteins]
            generated_tokens = self.model_ref.encode(ESMProtein(sequence=item['generated_sequence'], 
                                                        coordinates=item['structure_prompt']))
        else:
            input_tokens = [self.model.encode(protein) for protein in proteins]
            generated_tokens = self.model.encode(ESMProtein(sequence=item['generated_sequence'], 
                                                        coordinates=item['structure_prompt']))
        
        
        devices = set([t.device for t in input_tokens])
        if len(devices) > 1:
                raise AttributeError(f"Input tokens on multiple devices {devices}")
        sequence_lengths = [len(tokens) for tokens in input_tokens]
        assert len(set(sequence_lengths)) == 1
        
        if ref:
            batched_tokens = _stack_protein_tensors(
                    input_tokens, sequence_lengths, self.model_ref.tokenizers, self.model_ref.device
                )
        else:
            batched_tokens = _stack_protein_tensors(
                    input_tokens, sequence_lengths, self.model.tokenizers, self.model.device
                )
        
        if batched_tokens.coordinates is None:
            per_res_plddt = None
        else:
            # 1.0 if all coordinates at specific indices have valid non-nan values.
            per_res_plddt = batched_tokens.coordinates.isfinite().all(dim=-1).any(dim=-1).float()

        if ref:
            assert model_ref.device == batched_tokens.device
            with (torch.no_grad(),
                torch.autocast(enabled=True, device_type=torch.device(batched_tokens.device).type, dtype=torch.bfloat16)):
                output = self.model_ref.forward(
                        sequence_tokens=batched_tokens.sequence,
                        structure_tokens=batched_tokens.structure,
                        ss8_tokens=batched_tokens.secondary_structure,
                        sasa_tokens=batched_tokens.sasa,
                        function_tokens=batched_tokens.function,
                        residue_annotation_tokens=batched_tokens.residue_annotations,
                        average_plddt=torch.tensor(1.0, device=batched_tokens.device),
                        per_res_plddt=per_res_plddt,
                        structure_coords=batched_tokens.coordinates,
                        chain_id=None,
                        sequence_id=None,
                )
        else:
            assert model.device == batched_tokens.device
            with (torch.autocast(enabled=True, device_type=torch.device(batched_tokens.device).type, dtype=torch.bfloat16)):
                output = self.model.forward(
                        sequence_tokens=batched_tokens.sequence,
                        structure_tokens=batched_tokens.structure,
                        ss8_tokens=batched_tokens.secondary_structure,
                        sasa_tokens=batched_tokens.sasa,
                        function_tokens=batched_tokens.function,
                        residue_annotation_tokens=batched_tokens.residue_annotations,
                        average_plddt=torch.tensor(1.0, device=batched_tokens.device),
                        per_res_plddt=per_res_plddt,
                        structure_coords=batched_tokens.coordinates,
                        chain_id=None,
                        sequence_id=None,
                )
        
        log_prob = torch.log_softmax(output.sequence_logits, dim=-1)

        masked_log_prob = log_prob[:, item['masked_positions'] + 1, generated_tokens.sequence[item['masked_positions'] + 1]]
        pll = torch.sum(masked_log_prob)

        # print(item['masked_sequence'])
        print(generated_tokens.sequence[item['masked_positions'] + 1])
        print(masked_log_prob)
        # print(pll)

        return pll
         
    def dpo_paired_loss(self, batch):

        pll_policy = []
        pll_ref = []
        for item in batch:
            pll_policy.append(self.get_PLL(item))
            pll_ref.append(self.get_PLL(item, ref=True))
            print(item['property'])

        print([x['property'] for x in batch])
        print(pll_policy)
        print(pll_ref)

        pll_ref = [x.to(self.model.device) for x in pll_ref]

        print(pll_ref)

        batch_size = len(batch)
        assert batch_size > 1
        beta = 0.01

        loss = []

        pairs_debug = []
        for i in range(batch_size-1):
            for j in range(1, batch_size):
                if batch[i]['property'] > batch[j]['property'] + 0:
                    loss_item = -torch.nn.functional.logsigmoid( beta * ((pll_policy[i] - pll_ref[i]) - (pll_policy[j] - pll_ref[j])) )
                    # loss_item = -torch.nn.functional.logsigmoid( beta * ((pll_policy[i]) - (pll_policy[j])) )
                    loss.append(loss_item)

                    pairs_debug.append((batch[i]['property'], batch[j]['property'],
                                        (pll_policy[i] - pll_ref[i]).item(), (pll_policy[j] - pll_ref[j]).item(),
                                        loss_item.item()))
                    
                elif batch[j]['property'] > batch[i]['property'] + 0:
                    loss_item = -torch.nn.functional.logsigmoid( beta * ((pll_policy[j] - pll_ref[j]) - (pll_policy[i] - pll_ref[i])) )
                    # loss_item = -torch.nn.functional.logsigmoid( beta * ((pll_policy[j]) - (pll_policy[i])) )
                    loss.append(loss_item)

                    pairs_debug.append((batch[j]['property'], batch[i]['property'],
                                        (pll_policy[j] - pll_ref[j]).item(), (pll_policy[i] - pll_ref[i]).item(),
                                        loss_item.item()))
                    
        for _pair in pairs_debug:
            print(f'pair :{_pair}')

        if len(loss) > 0:
            loss = torch.mean(torch.stack(loss))
        else:
            loss = torch.tensor(0.).to(self.model.device)
                    
        return loss
 
    def train(self, sequence_prompt, structure_prompt):

        self.model.train()

        self.print_trainable_parameters(self.model)
        self.print_trainable_parameters(self.model_ref)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=1e-3,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=0.1,
        )

        mask_positions = self.get_mask_positions(sequence_prompt=sequence_prompt)

        epochs = 100
        batch_size = 1
        for epoch in range(epochs):
            optimizer.zero_grad()

            batch_dataset = []
            print(f'epoch: {epoch}: Generating batch')
            batch_seq = self.generate_batch(sequence_prompt=sequence_prompt, 
                                        structure_prompt=structure_prompt,
                                        batch_size=batch_size)
            
            for _seq in batch_seq:
                batch_dataset.append({
                    'generated_sequence': _seq,
                    'masked_sequence': ''.join([s for _i, s in enumerate(_seq) if _i in mask_positions]),
                    'masked_positions': mask_positions, ## These are the same
                    'sequence_prompt': sequence_prompt, ## These are the same
                    'structure_prompt': structure_prompt ## These are the same        
                })

            A, C = sequence_prompt.split('_'*115)
            base = 'APSEDAYPRATYEAPEGETEQLLAGIWMDLLQVDRVGRHDSFFELGGHSLLAVRLLGRLRQHGLGLQMRDLFEAPVLAELATRLRPYQPLEVPANGITPDTTVLTPEMLPLVTLS'
            batch_dataset.append({
                'generated_sequence': A+base+C,
                'masked_sequence': base,
                'masked_positions': mask_positions, ## These are the same
                'sequence_prompt': sequence_prompt, ## These are the same
                'structure_prompt': structure_prompt ## These are the same  
            })

            batch_dataset = self.get_property_batch(batch_dataset)
            
            print(f'epoch {epoch}: calculating loss')
            loss = self.dpo_paired_loss(batch_dataset)
            print(f'loss {loss.item()}')

            if loss != 0:
                print(f'epoch {epoch}: optimizer step')
                loss.backward()
                optimizer.step()

            torch.cuda.empty_cache()

model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda:0")
model_ref = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda:1")

protein = ProteinChain.from_pdb('../../Data/gxps/gxps_ATC_AF.pdb')

sequence_prompt = ''.join([protein[i].sequence if i in A_gxps_atc + C_gxps_atc else '_' for i in range(len(protein))])
# structure_prompt = torch.full((len(sequence_prompt), 37, 3), np.nan)
# structure_prompt[T_gxps_atc] = torch.tensor(protein.atom37_positions)[T_gxps_atc]
structure_prompt = torch.tensor(protein.atom37_positions)

# structure_prompt = torch.tensor(protein.atom37_positions)[T_gxps_atc[0]-150:T_gxps_atc[-1]+150]

# sequence_prompt = ''.join([s for i, s in enumerate(sequence_prompt) if i in list(range(T_gxps_atc[0]-150,T_gxps_atc[-1]+150))])

os.environ["DISABLE_ITERATIVE_SAMPLING_TQDM"] = "True"
dpo_trainer = DPOTrainer(model, model_ref, device='gpu')

dpo_trainer.train(sequence_prompt=sequence_prompt, structure_prompt=None)