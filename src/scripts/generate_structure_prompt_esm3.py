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

from DomainPrediction import BaseProtein
from DomainPrediction.eval import metrics
from DomainPrediction.utils import helper
from DomainPrediction.utils.constants import *

protein = ProteinChain.from_pdb('../../Data/gxps/gxps_ATC_AF.pdb')

sequence_prompt = ''.join([protein[i].sequence if i in A_gxps_atc + C_gxps_atc else '_' for i in range(len(protein))])
# structure_prompt = torch.full((len(sequence_prompt), 37, 3), np.nan)
# structure_prompt[T_gxps_atc] = torch.tensor(protein_T_domain.atom37_positions)
# structure_prompt[T_gxps_atc] = torch.tensor(protein.atom37_positions)[T_gxps_atc]
structure_prompt = torch.tensor(protein.atom37_positions)

model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

fasta_file = '../../Data/round_3_exp/esm3_str_2000.fasta' ## file loc

N_GENERATIONS = 2000
temperature = 0.5
run_structure = False
print(f'T domain: {protein[T_gxps_atc].sequence}')
for idx in range(N_GENERATIONS):
    
    # if run_structure and idx > 1:
    #     run_structure = False
    #     print('stopping structure prediction')

    sequence_prediction_config = GenerationConfig(
        track="sequence", 
        num_steps=sequence_prompt.count("_") // 2, 
        temperature=temperature
    )
    esm_protein = ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt)
    generated_protein = model.generate(esm_protein, sequence_prediction_config)

    # if run_structure:
    #     ## generate structure from the generated sequence
    #     structure_prediction_config = GenerationConfig(
    #         track="structure",
    #         num_steps=len(generated_protein) // 8,
    #         temperature=temperature, 
    #     )
    #     structure_prediction_prompt = ESMProtein(sequence=generated_protein.sequence)
    #     structure_prediction = model.generate(structure_prediction_prompt, structure_prediction_config)

    #     assert generated_protein.sequence == structure_prediction.sequence
    #     # structure_prediction.to_pdb(os.path.join(pdbfile_loc, gen_idx))

    print(f"T domain: {''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in T_gxps_atc])}")

    assert protein[A_gxps_atc].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in A_gxps_atc])
    assert protein[C_gxps_atc].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in C_gxps_atc])

    seq_dict = {}
    gen_idx = f'gxps_ATC_esm3_str_gen_{idx}'
    seq_dict[gen_idx] = generated_protein.sequence

    print(gen_idx)

    helper.create_fasta(seq_dict, fasta_file, append=True)