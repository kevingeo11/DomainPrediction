import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

import numpy as np
from tqdm import tqdm

class ESMCLM():
    def __init__(self, name, device='cpu') -> None:
        if name == 'esmc_300m':
            self.model = ESMC.from_pretrained(name)
            self.emb_dim = 960
        elif name == 'esmc_600m':
            self.model = ESMC.from_pretrained(name)
            self.emb_dim = 1152
        else:
            raise Exception('Check ESMC name')

        self.device = device
        if self.device == 'gpu':
            self.model = self.model.to("cuda")
        else:
            self.model = self.model.to("cpu")
        
        self.model.eval()

    def get_res(self, sequence, return_embeddings=False):
        esm_protein = ESMProtein(sequence=sequence)
        esm_tensor = self.model.encode(esm_protein)

        results = self.model.logits(
            esm_tensor, LogitsConfig(sequence=True, return_embeddings=return_embeddings)
        )

        return results

    def get_logits(self, sequence):

        results = self.get_res(sequence)
        logits = results.logits.sequence

        return logits

    def get_prob(self, sequence):
        logits = self.get_logits(sequence)
        prob = torch.nn.functional.softmax(logits[0, 1:-1, :33], dim=-1) # 33 token - esm3 outputs 64 tokens for optimazation

        return prob.to(torch.float32).cpu().numpy()
    
    def get_embeddings_mean(self, sequences):
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq, return_embeddings=True)
            embeddings.append(rep.embeddings[:,1:-1,:].mean(1).cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
    
    def get_embeddings_flatten(self, sequences):
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq, return_embeddings=True)
            embeddings.append(rep.embeddings[:,1:-1,:].cpu().numpy()[0].flatten())

        embeddings = np.stack(embeddings, axis=0)

        return embeddings
    
    def get_embeddings_feature_pool(self, sequences, pool='mean'):
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq, return_embeddings=True)
            if pool == 'mean':
                embeddings.append(rep.embeddings[:,1:-1,:].mean(-1).cpu().numpy())
            elif pool == 'sum':
                embeddings.append(rep.embeddings[:,1:-1,:].sum(-1).cpu().numpy())
            else:
                raise Exception('pool can only take values mean or sum')
            
        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
    
    def get_embeddings_cls(self, sequences):
        embeddings = []
        for seq in tqdm(sequences):
            rep = self.get_res(sequence=seq, return_embeddings=True)
            embeddings.append(rep.embeddings[:, 0, :].cpu().numpy())

        embeddings = np.concatenate(embeddings, axis=0)

        return embeddings
    
    def compute_perplexity(self, sequence, mask_token='_'):
        '''
            pseudoperplexity(x) = exp( -1/L \sum_{i=1}_{L} [log( p(x_{i}|x_{j!=i}) )] )
            we can use <mask> token also
        '''
        
        sum_log = 0
        for pos in range(len(sequence)):
            masked_query = list(sequence)
            assert mask_token not in masked_query
            masked_query[pos] = mask_token
            masked_query = ''.join(masked_query)
            prob = self.get_prob(sequence=masked_query)

            assert prob.shape[0] == len(sequence)

            prob_pos = np.log(prob[pos, self.model.tokenizer.convert_tokens_to_ids(sequence[pos])])
            
            sum_log += prob_pos

        return np.exp(-1*sum_log/len(sequence))
    
    def get_log_prob(self, sequence):
        logits = self.get_logits(sequence)
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
    
    def get_masked_marginal_var(self, mt_sequence, wt_sequence, mask_token = '_', mode='wt'):

        assert len(wt_sequence) == len(mt_sequence)

        n_muts = 0
        score = 0
        for i, (aa_mt, aa_wt) in enumerate(zip(mt_sequence, wt_sequence)):
            if aa_wt != aa_mt:
                ## mutation pos
                n_muts += 1

                masked_query_mt = list(mt_sequence)
                masked_query_mt[i] = mask_token
                masked_sequence_mt = ''.join(masked_query_mt)
                masked_log_prob_mt = self.get_log_prob(sequence=masked_sequence_mt)

                if mode == 'wt':
                    masked_query_wt = list(wt_sequence)
                elif mode == 'mt':
                    masked_query_wt = list(mt_sequence)
                else:
                    raise Exception('mode takes values mt and wt')

                masked_query_wt[i] = mask_token
                masked_sequence_wt = ''.join(masked_query_wt)
                masked_log_prob_wt = self.get_log_prob(sequence=masked_sequence_wt)

                idx_mt = self.model.tokenizer.convert_tokens_to_ids(aa_mt)
                idx_wt = self.model.tokenizer.convert_tokens_to_ids(aa_wt)
                score += masked_log_prob_mt[i, idx_mt] - masked_log_prob_wt[i, idx_wt]


        return score, n_muts
    
    def pseudolikelihood(self, mt_sequence, mask_token = '_'):

        score = 0
        for i, aa_mt in enumerate(zip(mt_sequence)):

            masked_query_mt = list(mt_sequence)
            masked_query_mt[i] = mask_token
            masked_sequence_mt = ''.join(masked_query_mt)
            masked_log_prob_mt = self.get_log_prob(sequence=masked_sequence_mt)

            idx_mt = self.model.tokenizer.convert_tokens_to_ids(aa_mt)
            score += masked_log_prob_mt[i, idx_mt]


        return score