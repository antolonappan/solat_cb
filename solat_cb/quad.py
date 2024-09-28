import numpy as np
import healpy as hp
import os
from tqdm import tqdm
from workspace.solat_cb.solat_cb.simulation.sky import LATsky
from solat_cb import mpi


class QE:
    def __init__(self,libdir,nside,alpha,dust,synch,beta):
        self.lat = LATsky(libdir,nside,alpha,dust,synch,beta)
        self.nside = nside
    

    