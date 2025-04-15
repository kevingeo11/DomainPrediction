import sys
sys.path.append('..')

import os
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from scipy import stats

from DomainPrediction.utils import helper
from DomainPrediction.utils.constants import *
from DomainPrediction.protein.base import BaseProtein

sys.path.append('/nethome/kgeorge/workspace/DomainPrediction/esm')
from DomainPrediction.esm.esmc import ESMCLM


data_path = '/nethome/kgeorge/workspace/DomainPrediction/Data/round_3_exp'
df_gen = pd.read_csv(os.path.join(data_path, 'gen_seq.csv'))

gxps_protein = BaseProtein(file='/nethome/kgeorge/workspace/DomainPrediction/Data/gxps/gxps_ATC_AF.pdb')
gxps_T_domain = ''.join([gxps_protein.sequence[i] for i in range(len(gxps_protein.sequence)) if i not in A_gxps_atc+C_gxps_atc])
gxps_base_seq = gxps_protein.sequence

assert len(gxps_T_domain) == 115

esmc = ESMCLM(name='esmc_600m', device='gpu')

y_pred = []
for i, row in df_gen.iterrows():

    masked_sequence = row['masked_sequence']
    print(f'{i}: {masked_sequence}')
    
    esmc_score_wt_marginal, n_muts = esmc.get_wildtype_marginal(masked_sequence, gxps_T_domain)
    assert n_muts == row['n_mut']
    esmc_score_masked_marginal, n_muts = esmc.get_masked_marginal(masked_sequence, gxps_T_domain)
    assert n_muts == row['n_mut']
    esmc_score_pll = esmc.pseudolikelihood(masked_sequence)[0]


    full_sequence = row['sequence']
    
    esmc_full_score_wt_marginal, n_muts = esmc.get_wildtype_marginal(full_sequence, gxps_base_seq)
    assert n_muts == row['n_mut']
    esmc_full_score_masked_marginal, n_muts = esmc.get_masked_marginal(full_sequence, gxps_base_seq)
    assert n_muts == row['n_mut']

    y_pred.append({
        'esmc_wt_marginal': esmc_score_wt_marginal,
        'esmc_masked_marginal': esmc_score_masked_marginal,
        'esmc_pll': esmc_score_pll,

        'esmc_full_wt_marginal': esmc_full_score_wt_marginal,
        'esmc_full_masked_marginal': esmc_full_score_masked_marginal,
    })

df_pred = pd.DataFrame(y_pred)
df_zero_shot_results = pd.concat([df_gen, df_pred], axis=1)
df_zero_shot_results.to_csv(os.path.join(data_path, 'gen_zero_shot_results_sys.csv'), index=False)