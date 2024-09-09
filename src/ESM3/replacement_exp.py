import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append('../../esm') ## ignore if you are installing esm3 and use huggingface_hub login()
sys.path.append('..')

import numpy as np
import random
import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain


protein = ProteinChain.from_pdb('../../Data/GxpS_ATC.pdb') ## location to pdb file

A = [i for i in range(33,522)] ## 34-522
C = [i for i in range(637,1067)] ## 638-1067
T = [i for i in range(538,608)] ## 539-608

sequence_prompt = ''.join([protein[i].sequence for i in range(len(protein))])

model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

# print('T domain: ', ''.join([sequence_prompt[i] for i in range(len(sequence_prompt)) if i in T]))

N_GENERATIONS = 100000
current_prompt = sequence_prompt
# print('n masks: ', current_prompt.count('_'))
for idx in range(N_GENERATIONS):
    # gen_idx = f'GxpS_ATC-temp_{0.5}-gen_{idx}'
    print(f'Round {idx} -----')

    sample_pos = random.sample(T, 1)[0]
    # print('Before masking: ', ''.join([current_prompt[i] for i in range(len(current_prompt)) if i in T]))
    current_prompt = current_prompt[:sample_pos] + '_' + current_prompt[sample_pos+1:]
    # print('After masking:  ', ''.join([current_prompt[i] for i in range(len(current_prompt)) if i in T]))

    assert current_prompt.count('_') == 1

    sequence_prediction_config = GenerationConfig(
        track="sequence", 
        num_steps=1, 
        temperature=0.5
    )
    esm_protein = ESMProtein(sequence=current_prompt)
    generated_protein = model.generate(esm_protein, sequence_prediction_config)

    generated_T = ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in T])
    # print("Gen T domain:   ", generated_T)

    assert len(generated_protein.sequence) == len(current_prompt)
    assert protein[A].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in A])
    assert protein[C].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in C])

    current_prompt = generated_protein.sequence

    with open('replacement_logs.txt', 'a') as f:
        f.write(generated_protein.sequence + '\n')