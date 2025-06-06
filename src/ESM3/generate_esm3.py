import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append('../../esm') ## ignore if you are installing esm3 and use huggingface_hub login()
sys.path.append('..')

import numpy as np
import torch

from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, GenerationConfig
from esm.utils.structure.protein_chain import ProteinChain

from DomainPrediction import BaseProtein
from DomainPrediction.eval import metrics
from DomainPrediction.utils import helper
from DomainPrediction.utils.constants import *


protein = ProteinChain.from_pdb('../../Data/6mfw_conformations/hm_6mfz_ATC.pdb')

sequence_prompt = ''.join([protein[i].sequence if i not in T_6mfw else '_' for i in range(len(protein))])
structure_prompt = torch.tensor(protein.atom37_positions)

model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

fasta_file = '../../Data/esm3_experiments/6mfw_exp/6mfw_esm3_1000.fasta' ## file loc

N_GENERATIONS = 1000
temperature = 0.5
run_structure = False
print(f'T domain: {protein[T_6mfw].sequence}')
for idx in range(N_GENERATIONS):
    
    if run_structure and idx > 1:
        run_structure = False
        print('stopping structure prediction')

    sequence_prediction_config = GenerationConfig(
        track="sequence", 
        num_steps=sequence_prompt.count("_") // 2, 
        temperature=temperature
    )
    esm_protein = ESMProtein(sequence=sequence_prompt)
    generated_protein = model.generate(esm_protein, sequence_prediction_config)

    if run_structure:
        ## generate structure from the generated sequence
        structure_prediction_config = GenerationConfig(
            track="structure",
            num_steps=len(generated_protein) // 8,
            temperature=temperature, 
        )
        structure_prediction_prompt = ESMProtein(sequence=generated_protein.sequence)
        structure_prediction = model.generate(structure_prediction_prompt, structure_prediction_config)

        assert generated_protein.sequence == structure_prediction.sequence
        # structure_prediction.to_pdb(os.path.join(pdbfile_loc, gen_idx))

    print(f"T domain: {''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in T_6mfw])}")

    assert protein[A_6mfw].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in A_6mfw])
    assert protein[C_6mfw].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in C_6mfw])

    seq_dict = {}
    gen_idx = f'6mfw_ATC_esm3_temp_{temperature}_gen_{idx}'
    seq_dict[gen_idx] = generated_protein.sequence

    print(gen_idx)

    helper.create_fasta(seq_dict, fasta_file, append=True)