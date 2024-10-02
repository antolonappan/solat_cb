import numpy as np
import sys
sys.path.append('/global/homes/l/lonappan/workspace/solat_cb')
from solat_cb import mpi
from solat_cb.simulation import LATsky
from solat_cb.spectra import Spectra
from solat_cb.mle import MLE

libdir ='/pscratch/sd/l/lonappan/SOLAT'
nside = 1024
cb_method = 'iso'
beta = 0.35
dust = 10
synch = 5
alpha = [-0.1,-0.1,0.2,0.2,.15,.15]
atm_noise = True
nsplits = 2
bandpass = False
fit = "As + Asd + Ad + beta + alpha"
binwidth = 20
bmin = 60
bmax = 2000

lat = LATsky(libdir,nside,cb_method,dust,synch,alpha,beta,atm_noise=atm_noise,nsplits=nsplits,bandpass=bandpass)
#spec = Spectra(lat,cache=True,parallel=1)
spec = Spectra(lat,cache=True,parallel=1,dust_model=9,sync_model=4)
mle = MLE(libdir,spec,fit, alpha_per_split=False,rm_same_tube=False,binwidth=binwidth,bmin=bmin,bmax=bmax)
jobs = np.arange(100)
for i in jobs[mpi.rank::mpi.size]:
    di = mle.estimate_angles(i)
mpi.barrier()