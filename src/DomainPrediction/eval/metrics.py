import os
from tqdm import tqdm
from Bio import AlignIO
import re
import numpy as np
from ..utils.hmmtools import HmmerTools
from ..utils import helper

def compute_sequence_identity(wt: str, gen: str, hmm: str, trim: bool=False) -> list:
    if len(helper.read_fasta(wt, mode='str')) == 0 or len(helper.read_fasta(gen, mode='str')) == 0:
        raise Exception('One of the fasta files is empty')
    
    if len(helper.read_fasta(wt, mode='str')) > 1:
        raise Exception('Sorry! Current functionality is limited to one WT sequence. Feel free to extend :D')
    
    ## need to select seq from gen and combine with wt to create a fasta file
    wt_record = helper.read_fasta(wt)[0]
    gen_records = helper.read_fasta(gen)

    hmmer = HmmerTools()
    seq_id_list = []
    for rec in tqdm(gen_records):
        tmp_file = os.path.join(os.path.dirname(gen), rec.id + '.tmp.fasta')
        helper.create_fasta(
            {
                wt_record.id : str(wt_record.seq),
                rec.id : str(rec.seq)
            },
            tmp_file
        )
        hmmer.hmmalign(hmm_path=hmm, fasta_file=tmp_file)
        alignment = AlignIO.read(tmp_file.replace('.fasta', '.stockholm'), "stockholm")
        alignment_length = alignment.get_alignment_length()

        assert len(alignment) == 2

        seq_id = 0
        ## should we treat caps?
        for s1, s2 in zip(alignment[0].seq, alignment[1].seq):
            if s1 != '-' and s2 != '-' and s1 != '.' and s2 != '.' and s1 == s2:
                seq_id += 1

        seq_id_list.append(seq_id/alignment_length)

        os.remove(tmp_file)
        os.remove(tmp_file.replace('.fasta', '.stockholm'))

    return seq_id_list


def search_and_filter_motif(file: str, pattern: str = 'FF.{2}GG.{1}S'):
    '''
        Extend to list of sequences or dict?
        Extend to save list to fasta or return selected sequences
    '''

    records = helper.read_fasta(file)

    matches = []
    for rec in records:
        if re.search(pattern, str(rec.seq)):
            matches.append(rec)
    
    print(f'{len(matches)*100/len(records)}% records contain motif')


def compute_perplexity(model, sequence, mask_token='<mask>'):
    '''
        pseudoperplexity(x) = exp( -1/L \sum_{i=1}_{L} [log( p(x_{i}|x_{j!=i}) )] )
    '''
    
    sum_log = 0
    for pos in tqdm(range(len(sequence))):
        masked_query = list(sequence)
        assert mask_token not in masked_query
        masked_query[pos] = mask_token
        masked_query = ''.join(masked_query)
        prob = model.get_prob(sequence=masked_query)

        assert prob.shape[0] == len(sequence)

        prob_pos = np.log(prob[pos, model.tok_to_idx[sequence[pos]]])
        
        sum_log += prob_pos

    return np.exp(-1*sum_log/len(sequence))

    