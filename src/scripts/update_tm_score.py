import sys
sys.path.append('..')

import os
from tqdm import tqdm
import numpy as np

from DomainPrediction.utils import helper
from DomainPrediction.utils.tmalign import TMalign

root = '../../..'
data_path = '/data/users/kgeorge/workspace/Data'
pmpnn_path = os.path.join(data_path, 'pmpnn_experiments/gxps_exp')
esm3_path = os.path.join(data_path, 'esm3_experiments/gxps_exp')
evodiff_path = os.path.join(data_path, 'evodiff_experiments/gxps_exp')
esm2_entropy_path = os.path.join(data_path, 'esm2_experiments/entropy/gxps_exp')

tm_path = '/nethome/kgeorge/workspace/DomainPrediction/src/DomainPrediction/utils/TMalign'
ref_paths = [os.path.join(data_path, f) for f in ['gxps/gxps_ATC_hm_6mfy.pdb', 'gxps/gxps_ATC_hm_6mfz.pdb', 
                                                  'gxps/gxps_ATC_hm_6mg0_A.pdb', 'gxps/gxps_ATC_hm_6mg0_B.pdb']]


tmalign = TMalign(tm_path)
paths = [pmpnn_path, esm3_path, evodiff_path, esm2_entropy_path]
for _path in paths:
    print(_path)
    pdbs_path = os.path.join(_path, 'gxps_pdbs')
    for f in os.listdir(pdbs_path):
        if f.endswith('.pdb') and not f.endswith('T.pdb'):
            print(f)
            file = os.path.join(pdbs_path, f)
            scores = []
            for ref_path in ref_paths: 
                res = tmalign.run(ref_path, file)
                scores.append(res['tm_score'])

            tm_score = max(scores)
            meta_file = file.replace('.pdb', '.meta.npz') ## need to change this
            helper.update_metadata(meta_file, 'max_TM_score', tm_score, force=False)
    
    ## Sanity check
    print('Checking keys in metadata')
    for f in os.listdir(pdbs_path):
        if f.endswith('.meta.npz'):
            meta_file = os.path.join(pdbs_path, f)
            metadata = dict(np.load(meta_file))
            for key in ['predicted_aligned_error', 'ptm', 'esm2_650M_perplexity', 'max_TM_score']:
                assert key in metadata


tm_path = '/nethome/kgeorge/workspace/DomainPrediction/src/DomainPrediction/utils/TMalign'
ref_path = os.path.join(data_path, 'gxps/gxps_T.pdb') 


tmalign = TMalign(tm_path)
paths = [pmpnn_path, esm3_path, evodiff_path, esm2_entropy_path]
for _path in paths:
    print(_path)
    pdbs_path = os.path.join(_path, 'gxps_pdbs')
    for f in os.listdir(pdbs_path):
        if f.endswith('T.pdb'):
            print(f)
            file = os.path.join(pdbs_path, f)
            res = tmalign.run(ref_path, file)

            tm_score = res['tm_score']
            meta_file = file.replace('.T.pdb', '.meta.npz') ## need to change this

            helper.update_metadata(meta_file, 'T_TM_score', tm_score, force=False)
    
    # Sanity check
    print('Checking keys in metadata')
    for f in os.listdir(pdbs_path):
        if f.endswith('.meta.npz'):
            meta_file = os.path.join(pdbs_path, f)
            metadata = dict(np.load(meta_file))
            for key in ['predicted_aligned_error', 'ptm', 'esm2_650M_perplexity', 'max_TM_score', 'T_TM_score']:
                assert key in metadata