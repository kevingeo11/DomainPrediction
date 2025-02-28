import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRANSFORMERS_CACHE"] = "/nethome/kgeorge/workspace/DomainPrediction/Data/esm3_experiments/mi_exp"
# os.environ["MPLCONFIGDIR"] = "/nethome/kgeorge/workspace/DomainPrediction/Data/esm3_experiments/mi_exp"
sys.path.append('../../esm') ## ignore if you are installing esm3 and use huggingface_hub login()
sys.path.append('..')

import numpy as np
import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain

from DomainPrediction.utils import helper

# A_PSG = [i for i in range(0, 954)]
# T_PSG = [i for i in range(968, 1039)]
# C_PSG = [i for i in range(1057, 1489)]

A_star_pCK = [i for i in range(0, 485)]
T_star_pCK = [i for i in range(485, 610)]
C_star_pCK = [i for i in range(610, 1017)]

# protein = ProteinChain.from_pdb('/nethome/kgeorge/workspace/DomainPrediction/Data/new_system/PSG_KG.pdb')
protein = ProteinChain.from_pdb('/nethome/kgeorge/workspace/DomainPrediction/Data/new_system/pCK_KG.pdb')

# sequence_prompt = ''.join([protein[i].sequence if i in A_PSG + C_PSG else '_' for i in range(len(protein))])
sequence_prompt = ''.join([protein[i].sequence if i in A_star_pCK + C_star_pCK else '_' for i in range(len(protein))])
structure_prompt = torch.tensor(protein.atom37_positions)

model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

# fasta_file = '/nethome/kgeorge/workspace/DomainPrediction/Data/new_system/PSG_KG_esm3_str_1000.fasta'
fasta_file = '/nethome/kgeorge/workspace/DomainPrediction/Data/new_system/pCK_KG_esm3_str_1000.fasta'

N_GENERATIONS = 1000
temperature = 0.5
run_structure = False
# print(f'T domain: {protein[T_PSG].sequence}')
print(f'T domain: {protein[T_star_pCK].sequence}')
for idx in range(N_GENERATIONS):

    sequence_prediction_config = GenerationConfig(
        track="sequence", 
        num_steps=sequence_prompt.count("_") // 2, 
        temperature=temperature
    )
    esm_protein = ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt)
    generated_protein = model.generate(esm_protein, sequence_prediction_config)

    # print(f"T domain: {''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in T_PSG])}")
    print(f"T domain: {''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in T_star_pCK])}")

    # assert protein[A_PSG].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in A_PSG])
    # assert protein[C_PSG].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in C_PSG])
    assert protein[A_star_pCK].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in A_star_pCK])
    assert protein[C_star_pCK].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in C_star_pCK])

    seq_dict = {}
    # gen_idx = f'PSG_KG_esm3_str_gen_{idx}'
    gen_idx = f'pCK_KG_esm3_str_gen_{idx}'
    seq_dict[gen_idx] = generated_protein.sequence

    print(gen_idx)

    helper.create_fasta(seq_dict, fasta_file, append=True)