import sys
sys.path.append('..')

from DomainPrediction import BaseProtein
from DomainPrediction.utils import helper
from DomainPrediction.utils.constants import *
from DomainPrediction.esm.esm2 import ESM2_iterate

model_path = '/data/users/kgeorge/workspace/esm2/checkpoints/esm2_t33_650M_UR50D.pt'
esm2_iterate = ESM2_iterate(model_path=model_path, device='gpu')

protein = BaseProtein(file='../../Data/6mfw_conformations/hm_6mfz_ATC.pdb')

fasta_file = '../../Data/esm2_experiments/entropy/6mfw_exp/6mfw_esm2_entropy_1000.fasta' ## file loc

N_GEN = 1000
for i in range(N_GEN):
    # generated_sequence = esm2_iterate.random_fill(protein.sequence, infill_idx=T_6mfw, fix=1)
    generated_sequence = esm2_iterate.entropy_fill(protein.sequence, infill_idx=T_6mfw, fix=1)

    seq_dict = {}
    gen_idx = f'6mfw_ATC_esm2_entropy_gen_{i}'
    seq_dict[gen_idx] = generated_sequence

    print(gen_idx)

    helper.create_fasta(seq_dict, fasta_file, append=True)