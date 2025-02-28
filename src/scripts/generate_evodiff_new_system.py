import sys
sys.path.append('..')

import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from evodiff.pretrained import OA_DM_38M, OA_DM_640M
from evodiff.generate import generate_oaardm
from evodiff.conditional_generation import inpaint_simple 

torch.hub.set_dir('/data/users/kgeorge/workspace/evodiff')

from DomainPrediction import BaseProtein
from DomainPrediction.eval import metrics
from DomainPrediction.utils import helper

# A_PSG = [i for i in range(0, 954)]
# T_PSG = [i for i in range(968, 1039)]
# C_PSG = [i for i in range(1057, 1489)]

A_star_pCK = [i for i in range(0, 485)]
T_star_pCK = [i for i in range(485, 610)]
C_star_pCK = [i for i in range(610, 1017)]

# protein = BaseProtein(file='/nethome/kgeorge/workspace/DomainPrediction/Data/new_system/PSG_KG.pdb')
protein = BaseProtein(file='/nethome/kgeorge/workspace/DomainPrediction/Data/new_system/pCK_KG.pdb')

# start_idx, end_idx = A_PSG[-1]+1, C_PSG[0]
start_idx, end_idx = A_star_pCK[-1]+1, C_star_pCK[0]
start_idx, end_idx

idr_length = end_idx - start_idx
masked_sequence = protein.sequence[0:start_idx] + '#' * idr_length + protein.sequence[end_idx:]

# masked_sequence.count('#') == 103
masked_sequence.count('#') == 125

checkpoint = OA_DM_640M()
model, collater, tokenizer, scheme = checkpoint

model = model.cuda()

sequence = protein.sequence

# fasta_file = '/nethome/kgeorge/workspace/DomainPrediction/Data/new_system/PSG_KG_evodiff_1000.fasta'
fasta_file = '/nethome/kgeorge/workspace/DomainPrediction/Data/new_system/pCK_KG_evodiff_1000.fasta'
for i in range(1000):
    seq_dict = {}
    sample, entire_sequence, generated_idr = inpaint_simple(model, sequence, start_idx, end_idx, tokenizer=tokenizer, device='cuda')
    id = f'pCK_KG_evodiff_gen_{i}'
    seq_dict[id] = entire_sequence

    print(id)
    helper.create_fasta(seq_dict, fasta_file, append=True)
