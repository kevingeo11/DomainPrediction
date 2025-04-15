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
from DomainPrediction.utils.constants import *

protein = BaseProtein(file='/nethome/kgeorge/workspace/DomainPrediction/Data/gxps/gxps_ATC_AF.pdb')

start_idx, end_idx = A_gxps_atc[-1]+1, C_gxps_atc[0]
start_idx, end_idx

idr_length = end_idx - start_idx
masked_sequence = protein.sequence[0:start_idx] + '#' * idr_length + protein.sequence[end_idx:]

assert masked_sequence.count('#') == 115

checkpoint = OA_DM_640M()
model, collater, tokenizer, scheme = checkpoint

model = model.cuda()

sequence = protein.sequence

fasta_file = '/nethome/kgeorge/workspace/DomainPrediction/Data/round_3_exp/evodiff_5000.fasta'
for i in range(5000):
    seq_dict = {}
    sample, entire_sequence, generated_idr = inpaint_simple(model, sequence, start_idx, end_idx, tokenizer=tokenizer, device='cuda')
    id = f'gxps_ATC_evodiff_gen_{i}'
    seq_dict[id] = entire_sequence

    print(id)
    helper.create_fasta(seq_dict, fasta_file, append=True)