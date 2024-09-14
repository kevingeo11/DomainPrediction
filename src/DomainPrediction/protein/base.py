import Bio
import Bio.PDB
import os
import numpy as np

from .constants import AA20_3_TO_1

class BaseProtein():
    '''
        init with a sequence??
    '''
    def __init__(self, file : str, id: str = 'default') -> None:
        pdbparser = Bio.PDB.PDBParser(QUIET=True)
        self.struct = pdbparser.get_structure(id, file)

        n_chains = 0
        for chain in self.struct.get_chains():
            n_chains += 1

        if n_chains > 1:
            raise Exception('Method not designed for multiple chains')

        self.id = id
        self.sequence = ''.join([AA20_3_TO_1[res.resname] for res in chain.get_residues()])

    def get_residues(self, resnums: list):
        '''
            resnums starts from 0
        '''
        return ''.join([self.sequence[i] for i in resnums])
    

class FoldedProtein(BaseProtein):
    def __init__(self, file : str, id: str = 'default') -> None:
        super().__init__(file, id)
        
        self.plddts = np.array([a.get_bfactor() for a in self.struct.get_atoms()])
        self.plddt = self.plddts.mean()
        self.pTM = None
        self.pAE = None

        if os.path.isfile(file.replace('.pdb', '.meta.npz')):
            metadata = np.load(file.replace('.pdb', '.meta.npz'))
            self.pTM = metadata['ptm']
            self.pAE = metadata['predicted_aligned_error']
            self.metadata = dict(metadata)
            


class ESMBaseProtein(BaseProtein):
    def __init__(self, file: str, id: str = 'default') -> None:
        super().__init__(file, id)

    def get_masked_sequence(self, mask: list):

        return ''.join([self.sequence[i] if i in mask else '_' for i in range(len(self.sequence))])