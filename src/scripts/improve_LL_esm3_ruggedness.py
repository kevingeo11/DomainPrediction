import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append('../../esm') ## ignore if intsalling esm3
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import torch
import json
import pickle
import random
import logging

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig, SamplingConfig, LogitsConfig, SamplingTrackConfig
from esm.utils.structure.protein_chain import ProteinChain

from DomainPrediction.protein.base import BaseProtein
from DomainPrediction.utils.constants import *
from DomainPrediction.utils import helper

data_path = '../../Data/round_2_exp/ll_guidance_ruggedness'

logger = logging.getLogger('LL')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(data_path, f'ruggedness_ESM3_2_peak.log'))
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

records = helper.read_fasta('../../Data/round_2_exp/ll_guidance_ruggedness/ll_guidance.fasta')
for record in records:
    if record.id == 'll_guidance-ESM2':
        break

protein = BaseProtein(sequence=str(record.seq), id=record.id)

peak_sequence = protein.sequence

logger.info(f'starting rec id {record.id}')

model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

def calculate_likelihood(sequence):
    log_prob = 0
    for pos in range(len(sequence)):
        aa = sequence[pos]
        tokens = model.tokenizers.sequence.encode(aa)
        assert len(tokens) == 3
        token = tokens[1]
        assert model.tokenizers.sequence.decode(token) == aa

        sequence_prompt = sequence[:pos] + '_' + sequence[pos+1:]
        esm_protein = ESMProtein(sequence=sequence_prompt)
        protein_tensor = model.encode(esm_protein)
        res = model.logits(protein_tensor, LogitsConfig(
            sequence = True
        ))

        logits = res.logits.sequence[0, 1:-1, :][pos, :].cpu()
        prob = torch.nn.functional.softmax(logits, dim=0)
        log_prob += np.log(prob[token].numpy())

    return log_prob

# starting_sequence = WT_sequence

filename = os.path.join(data_path, 'likelihood_db.json')
with open(filename) as f:
    likelihood_db = json.load(f)

if peak_sequence not in likelihood_db:
    print('peak_sequence sequence not in DB')
    likelihood_db[peak_sequence] = calculate_likelihood(peak_sequence)

logger.info(f'peak_sequence sequence LL: {likelihood_db[peak_sequence]}')

## Get pos of Linker + T domain
positions = [i for i in range(len(peak_sequence)) if i not in A_gxps_atc + C_gxps_atc]
random.seed(0)
exp_results = {}
for n_mut_pos in range(3):

    logger.info(f'Start Run for # mutations: {n_mut_pos+1}')

    starting_sequence = peak_sequence

    m_tag = f'M{n_mut_pos+1}'
    exp_results[m_tag] = {}

    for _reps in range(3):

        rep_tag = m_tag + f'R{_reps+1}'
        exp_results[m_tag][rep_tag] = {}
        exp_results[m_tag][rep_tag]['n_mut'] = n_mut_pos+1
        exp_results[m_tag][rep_tag]['n_rep'] = _reps+1

        mut_pos = random.sample(positions, n_mut_pos+1)
        starting_sequence_list = list(starting_sequence)
        for _pos in mut_pos:
            aa = random.choice(amino_acid_alphabet_list)
            starting_sequence_list[_pos] = aa
        starting_sequence = ''.join([i for i in starting_sequence_list])

        _peak_seq = ''.join([peak_sequence[i] for i in range(len(peak_sequence)) if i not in A_gxps_atc + C_gxps_atc])
        logger.info(f'{rep_tag} - peak seq: {_peak_seq}')
        exp_results[m_tag][rep_tag]['peak_seq'] = peak_sequence
        _mut_seq = ''.join([starting_sequence[i] for i in range(len(starting_sequence)) if i not in A_gxps_atc + C_gxps_atc])
        logger.info(f'{rep_tag} - mutated seq (# mut {n_mut_pos+1}): {_mut_seq}')
        exp_results[m_tag][rep_tag]['mut_seq'] = starting_sequence
        exp_results[m_tag][rep_tag]['mutations'] = {}
        for _pos in mut_pos:
            logger.info(f'{rep_tag} - Mutations: At pos {_pos} (rel pos {_pos-positions[0]}) from WT {peak_sequence[_pos]} to MUT {starting_sequence[_pos]}')
            exp_results[m_tag][rep_tag]['mutations'][_pos] = (peak_sequence[_pos], starting_sequence[_pos])

        if starting_sequence not in likelihood_db:
            likelihood_db[starting_sequence] = calculate_likelihood(starting_sequence)
        max_LL = likelihood_db[starting_sequence]
        
        best_seq = starting_sequence

        logger.info(f'{rep_tag} - Starting LL guidance')
        for _j in range(150):
            logger.info(f'{rep_tag} - starting run {_j}: likelihood {max_LL}')
            _best_seq = ''.join([best_seq[i] for i in range(len(best_seq)) if i not in A_gxps_atc + C_gxps_atc])
            logger.info(f'{rep_tag} - best seq at start run {_j}: {_best_seq}')
            check_ll = max_LL
            for _i, pos in enumerate(positions):
                aa_wt = starting_sequence[pos]
                sequence_prompt = starting_sequence[:pos] + '_' + starting_sequence[pos+1:]

                assert sequence_prompt[pos] == '_'
                assert ''.join([sequence_prompt[i] for i in range(len(starting_sequence)) if i not in A_gxps_atc + C_gxps_atc])[_i] == '_'

                esm_protein = ESMProtein(sequence=sequence_prompt)
                protein_tensor = model.encode(esm_protein)
                res = model.logits(protein_tensor, LogitsConfig(
                    sequence = True
                ))

                assert res.logits.sequence.shape[1] == len(sequence_prompt) + 2

                logits = res.logits.sequence[0, 1:-1, :][pos, :].cpu()
                token_mt = torch.argmax(logits)
                aa_mt = model.tokenizers.sequence.decode(token_mt)
                gen_seq = starting_sequence[:pos] + aa_mt + starting_sequence[pos+1:]

                if gen_seq not in likelihood_db:
                    likelihood_db[gen_seq] = calculate_likelihood(gen_seq)
                
                ll = likelihood_db[gen_seq]

                # print(pos, aa_wt, aa_mt, ll)

                if ll > max_LL:
                    logger.info(f'{rep_tag} - mutation at pos {pos} (rel pos {_i}) from {aa_wt} to {aa_mt} improved ll from {max_LL} to {ll}')
                    max_LL = ll
                    best_seq = gen_seq
            
            logger.info(f'{rep_tag} - run {_j} end: likelihood improved from {check_ll} to {max_LL}')
            _best_seq = ''.join([best_seq[i] for i in range(len(best_seq)) if i not in A_gxps_atc + C_gxps_atc])
            logger.info(f'{rep_tag} - best seq at end run {_j}: {_best_seq}')

            _ini_seq = ''.join([starting_sequence[i] for i in range(len(starting_sequence)) if i not in A_gxps_atc + C_gxps_atc])
            assert len(_best_seq) == len(_ini_seq)
            for i, (aa_i, aa_j) in enumerate(zip(_ini_seq, _best_seq)):
                if aa_i != aa_j:
                    logger.info(f'{rep_tag} - mutation at rel {i} from {aa_i} to {aa_j}')

            exp_results[m_tag][rep_tag][f'round_{_j+1}_seq'] = best_seq

            if max_LL > check_ll:
                assert best_seq != starting_sequence
                starting_sequence = best_seq
            else:
                logger.info(f'{rep_tag} - No improvement in LL: prev {check_ll} curr {max_LL}')
                if starting_sequence == best_seq:
                    logger.info(f'{rep_tag} - Convergence')
                    exp_results[m_tag][rep_tag]['converged'] = True
                    exp_results[m_tag][rep_tag]['rounds'] = _j+1
                    exp_results[m_tag][rep_tag]['best_seq'] = best_seq
                    break
                else:
                    logger.info(f'{rep_tag} - start seq: {starting_sequence}')
                    logger.info(f'{rep_tag} - best seq: {best_seq}')
                    raise Exception('Starting and best seq not same with no improvement in LL. Need to check!')

            filename = os.path.join(data_path, 'likelihood_db.json')
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(likelihood_db, f, ensure_ascii=False, indent=4)

        # after convergence save
        filename = os.path.join(data_path, 'likelihood_db.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(likelihood_db, f, ensure_ascii=False, indent=4)

        filename = os.path.join(data_path, 'ruggedness_data_ll_guidance-ESM2.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(exp_results, f, ensure_ascii=False, indent=4)

        logger.info(f'{rep_tag} - END: peak_sequence sequence LL: {likelihood_db[peak_sequence]}')
        logger.info(f'{rep_tag} - END: best_sequence sequence LL: {likelihood_db[best_seq]}')