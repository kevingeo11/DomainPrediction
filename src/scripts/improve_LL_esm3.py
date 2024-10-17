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
import logging

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig, SamplingConfig, LogitsConfig, SamplingTrackConfig
from esm.utils.structure.protein_chain import ProteinChain

from DomainPrediction.protein.base import BaseProtein
from DomainPrediction.utils.constants import *
from DomainPrediction.utils import helper

data_path = '../../Data/round_2_exp'

logger = logging.getLogger('LL')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(os.path.join(data_path, f'LL_improv.log'))
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)

protein = BaseProtein('../../Data/gxps/gxps_ATC_hm_6mfy.pdb')

WT_sequence = protein.sequence

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

starting_sequence = WT_sequence

filename = os.path.join(data_path, 'likelihood_db.json')
with open(filename) as f:
    likelihood_db = json.load(f)

if starting_sequence not in likelihood_db:
    print('starting sequence not in DB')
    likelihood_db[starting_sequence] = calculate_likelihood(starting_sequence)

logger.info(f'starting sequence LL: {likelihood_db[starting_sequence]}')

max_LL = likelihood_db[starting_sequence]
positions = [i for i in range(len(starting_sequence)) if i not in A_gxps_atc + C_gxps_atc]
best_seq = starting_sequence

fasta_file = os.path.join(data_path, 'll_improve.fasta')
helper.create_fasta({
    'start_seq': best_seq
}, file=fasta_file, append=True)

for _j in range(150):
    logger.info(f'starting run {_j}: likelihood {max_LL}')
    _best_seq = ''.join([best_seq[i] for i in range(len(best_seq)) if i not in A_gxps_atc + C_gxps_atc])
    logger.info(f'best seq at start run {_j}: {_best_seq}')
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

        print(pos, aa_wt, aa_mt, ll)

        if ll > max_LL:
            logger.info(f'mutation at pos {pos} (rel pos {_i}) from {aa_wt} to {aa_mt} improved ll from {max_LL} to {ll}')
            max_LL = ll
            best_seq = gen_seq
    
    logger.info(f'run {_j} end: likelihood improved from {check_ll} to {max_LL}')
    _best_seq = ''.join([best_seq[i] for i in range(len(best_seq)) if i not in A_gxps_atc + C_gxps_atc])
    logger.info(f'best seq at end run {_j}: {_best_seq}')

    _ini_seq = ''.join([starting_sequence[i] for i in range(len(starting_sequence)) if i not in A_gxps_atc + C_gxps_atc])
    assert len(_best_seq) == len(_ini_seq)
    for i, (aa_i, aa_j) in enumerate(zip(_ini_seq, _best_seq)):
        if aa_i != aa_j:
            logger.info(f'mutation at rel {i} from {aa_i} to {aa_j}')

    helper.create_fasta({
        f'round_{_j+1}_seq': best_seq
    }, file=fasta_file, append=True)

    if max_LL > check_ll:
        assert best_seq != starting_sequence
        starting_sequence = best_seq
    else:
        logger.info(f'No improvement in LL: prev {check_ll} curr {max_LL}')
        if starting_sequence == best_seq:
            logger.info('Convergence')
            break
        else:
            logger.info(f'start seq: {starting_sequence}')
            logger.info(f'best seq: {best_seq}')
            raise Exception('Starting and best seq not same with no improvement in LL. Need to check!')

    filename = os.path.join(data_path, 'likelihood_db.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(likelihood_db, f, ensure_ascii=False, indent=4)