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

fasta_builder_dict = dict()
pdbfile_loc = '/nethome/kgeorge/workspace/DomainPrediction/Data/esm_experiments/basic_gen/pdbs'

protein = ProteinChain.from_pdb('../../Data/GxpS_ATC.pdb') ## location to pdb file

fasta_builder_dict['GxpS_ATC'] = protein.sequence

A = [i for i in range(33,522)] ## 34-522
C = [i for i in range(637,1067)] ## 638-1067
T = [i for i in range(538,608)] ## 539-608
cond = A + C

sequence_prompt = ''.join([protein[i].sequence if i in cond else '_' for i in range(len(protein))])
structure_prompt = torch.tensor(protein.atom37_positions)

model: ESM3InferenceClient = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")

fasta_file = '/nethome/kgeorge/workspace/DomainPrediction/Data/esm_experiments/gen_1000/esm_inp_seq_1000.fasta' ## file loc

N_GENERATIONS = 1000
run_structure = False
print(f'T domain: {protein[T].sequence}')
for idx in range(1894,2000):
    fout = open(fasta_file, 'a')
    gen_idx = f'GxpS_ATC-temp_{0.5}-gen_{idx}'
    print(gen_idx)

    if run_structure and idx > 1:
        run_structure = False
        print('stopping structure prediction')

    sequence_prediction_config = GenerationConfig(
        track="sequence", 
        num_steps=sequence_prompt.count("_") // 2, 
        temperature=0.5
    )
    # esm_protein = ESMProtein(sequence=sequence_prompt, coordinates=structure_prompt)
    esm_protein = ESMProtein(sequence=sequence_prompt) ## for seq as input
    generated_protein = model.generate(esm_protein, sequence_prediction_config)

    if run_structure:
        ## generate structure from the generated sequence
        structure_prediction_config = GenerationConfig(
            track="structure",
            num_steps=len(generated_protein) // 8,
            temperature=0.5, 
        )
        structure_prediction_prompt = ESMProtein(sequence=generated_protein.sequence)
        structure_prediction = model.generate(structure_prediction_prompt, structure_prediction_config)

        assert generated_protein.sequence == structure_prediction.sequence
        structure_prediction.to_pdb(os.path.join(pdbfile_loc, gen_idx))

    print(f"T domain: {''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in T])}")

    assert protein[A].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in A])
    assert protein[C].sequence == ''.join([generated_protein.sequence[i] for i in range(len(generated_protein.sequence)) if i in C])

    fasta_builder_dict[gen_idx] = generated_protein.sequence

    fout.write(f'>{gen_idx}\n')
    fout.write(generated_protein.sequence + '\n')
    fout.close()

# fasta_file = '/nethome/kgeorge/workspace/DomainPrediction/Data/esm_experiments/basic_gen/esm_inp_seq_str_gen_gpu.fasta' ## file loc
# with open(fasta_file, 'w') as fout:
#     for key in fasta_builder_dict:
#         fout.write(f'>{key}\n')
#         fout.write(fasta_builder_dict[key] + '\n')