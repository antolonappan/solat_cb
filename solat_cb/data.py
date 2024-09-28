import requests
from dataclasses import dataclass
from typing import Any, Optional
from solat_cb import mpi
import os
from tqdm import tqdm
from healpy import read_map
from pickle import load



@dataclass
class Data:
    filename: str
    directory: Optional[str] = None

    def __dir___(self) -> str:
        assert self.directory is not None, 'Directory is not set.'
        return os.path.join(self.directory, 'Data')

    @property
    def fname(self) -> str:
        directory = self.__dir___()
        return os.path.join(directory, self.filename)

    @property
    def url(self) -> str:
        return f"https://github.com/antolonappan/solat_cb/releases/download/1.0/{self.filename}"
    
    def __load__(self, fname: str) -> Any:
        ext = fname.split('.')[-1]
        match ext:
            case 'fits':
                return read_map(fname)
            case 'pkl':
                return load(open(fname, 'rb'))
            case 'ini':
                return None
            case _:
                raise ValueError(f'Unknown file extension: {ext}')

    @property
    def data(self) -> Any:
        fname = self.fname
        if os.path.isfile(fname):
            return self.__load__(fname)
        else:
            if mpi.rank == 0:
                os.makedirs(self.__dir___(), exist_ok=True)
                download_file(self.url, fname)
            mpi.barrier()
            return self.__load__(fname)
        

SAT_MASK = Data('binary_SAT_mask_N1024.fits')
LAT_MASK = Data('binary_LAT_mask_N1024.fits')
CO_MASK = Data('binary_CO_mask_N1024.fits')
PS_MASK = Data('binary_comb_PS_mask_N1024.fits')
BP_PROFILE = Data('bp_profile.pkl')
CAMB_INI = Data('cb.ini')