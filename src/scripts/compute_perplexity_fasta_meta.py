import sys
sys.path.append('..')

import os
from tqdm import tqdm

from DomainPrediction.esm.esm2 import ESM2
from DomainPrediction.eval import metrics
from DomainPrediction.utils import helper

model_path = '/data/users/kgeorge/workspace/esm2/checkpoints/esm2_t33_650M_UR50D.pt'
esm2 = ESM2(model_path = model_path, device='gpu')

fasta_path = '../../Data/round_2_exp/ll_guidance/ll_guidance.fasta'
meta_file = '../../Data/round_2_exp/ll_guidance/ll_guidance_metadata.json'

records = helper.read_fasta(fasta_path)
for rec in records:
    perplexity = metrics.compute_perplexity(esm2, str(rec.seq))

    print(rec.id, perplexity)
    helper.update_metadata_json(meta_file, rec.id, 'esm2_650M_perplexity', perplexity, force=False)