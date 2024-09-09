import os
from typing import List, Union

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def create_fasta(sequences: dict | List[SeqRecord], 
                 file: str = 'default.fasta', append: bool = False):
    if os.path.isfile(file) and not append:
        raise Exception(f'fasta file {file} exists. If you want to append try with append=True')
    
    if type(sequences) is dict:
        records = []
        for id in sequences:
            records.append(SeqRecord(Seq(sequences[id]), id=id, description=""))
    else:
        assert type(sequences[0]) is type(SeqRecord)
        records = sequences

    if append:
        with open(file, 'a') as file:
            SeqIO.write(records, file, "fasta")
    else:
        with open(file, 'w') as file:
            SeqIO.write(records, file, "fasta")


def read_fasta(file: str, mode: str='default') -> List[type[SeqRecord] | str]:
    records = SeqIO.parse(file, "fasta")

    if mode == 'str':
        return [str(i.seq) for i in records]
    else:
        return [i for i in records]