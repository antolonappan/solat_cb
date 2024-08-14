import numpy as np
import sys
sys.path.append('/global/homes/l/lonappan/workspace/solat_cb')
from lat_cb import mpi
from lat_cb.mle import MLE

libdir ='/pscratch/sd/l/lonappan/SOLAT'
nside = 512
alpha = 0.35
dust = 1
synch = 1
beta = [0.1,0.1,0.8,0.8,.2,.2]
lmax = 1000
mle = MLE(libdir,nside,alpha,dust,synch,beta,lmax)

jobs = np.arange(100)
for i in jobs[mpi.rank::mpi.size]:
    di = mle.estimate_angle(i)
mpi.barrier()