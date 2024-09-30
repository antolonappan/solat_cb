# object oriented version of Patricia's code
import numpy as np
import healpy as hp
import os
import pickle as pl
from lat_cb.spectra import Spectra
from lat_cb.signal import CMB, LATsky
from lat_cb import mpi

rad2arcmin = 180*60/np.pi

def effective_ell(lmax, bin_conf):
    (ib_grid, il_grid, w_array, ell_array) = bin_conf
    ell = np.arange(0, lmax+1, 1)
    return np.sum(w_array[ib_grid,il_grid]*ell[ell_array[ib_grid,il_grid]], axis=1)

def bin_from_edges(start, end):
    nls  = np.amax(end)
    lmax = nls-1
    ells, bpws, weights = [], [], []
    for ib, (li, le) in enumerate(zip(start, end)):
        nlb      = int(le - li)
        ells    += list(range(li, le))
        bpws    += [ib] * nlb
        weights += [1./nlb] * nlb
    ells = np.array(ells); bpws = np.array(bpws); weights = np.array(weights)

    nell       = len(ells)
    n_bands    = bpws[-1]+1
    nell_array = np.zeros(n_bands, dtype=int)
    for ii in range(0, nell, 1):
        if ells[ii]<=lmax and bpws[ii]>=0:
            nell_array[bpws[ii]]+=1   
            
    ell_list, w_list = [], []
    for ii in range(0, n_bands, 1):
        ell_list.append(np.zeros(nell_array[ii], dtype=int))
        w_list.append(np.zeros(nell_array[ii], dtype=np.float64))
    
    nell_array *= 0
    for ii in range(0, nell, 1):
        if ells[ii]<=lmax and bpws[ii]>=0: 
            ell_list[bpws[ii]][nell_array[bpws[ii]]] = ells[ii]
            w_list[bpws[ii]][nell_array[bpws[ii]]]   = weights[ii]
            nell_array[bpws[ii]]+=1

    for ii in range(0, n_bands, 1):
        norm=0
        for jj in range(0, nell_array[ii], 1):
            norm += w_list[ii][jj]
        if norm<=0:
            print(f"Weights in band {ii} are wrong\n")
        for jj in range(0, nell_array[ii], 1):
            w_list[ii][jj]/=norm

    return (n_bands, nell_array, ell_list, w_list)

def bin_configuration(info):
    (n_bands, nell_array, ell_list, w_list) = info
    ib_array          = np.arange(0, n_bands,       1, dtype=int)
    il_array          = np.arange(0, nell_array[0], 1, dtype=int)
    (ib_grid,il_grid) = np.meshgrid(ib_array, il_array, indexing='ij')
    return (ib_grid, il_grid, np.array(w_list), np.array(ell_list))

#TODO might change the shape of the output array when optimising linear system terms
def bin_spec_matrix(spec, info):
    (ib_grid, il_grid, w_array, ell_array) = info
    return np.sum(w_array[ib_grid,il_grid]*spec[:,:,ell_array[ib_grid,il_grid]], axis=3)

#TODO might change the shape of the output array when optimising linear system terms
def bin_cov_matrix(cov, info):
    (ib_grid, il_grid, w_array, ell_array) = info
    return np.sum(w_array[ib_grid,il_grid]**2*cov[:,:,:,ell_array[ib_grid,il_grid]], axis=4)

def moving_sum(a, n) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] 


class Result:
    
    def __init__(self, spec, fit, sim, 
                 alpha_per_split, rm_same_tube,
                 binwidth, bmin, bmax,
                 beta_ini=0.0, alpha_ini=0.0):

        # information I should record to remember how the result was calculated
        # information about the spectra
        self.specdir  = spec.lat.libdir
        self.temp_bp  = spec.temp_bp
        self.aposcale = spec.aposcale
        self.CO       = spec.CO
        self.PS       = spec.PS
        self.pureB    = spec.pureB
        self.nside    = spec.nside
        self.fsky     = spec.fsky
        self.sim_idx  = sim 
        # information about the fit
        self.nlb             = binwidth
        self.bmin            = bmin
        self.bmax            = bmax
        self.fit             = fit
        self.rm_same_tube    = rm_same_tube
        self.alpha_per_split = alpha_per_split
        # parameters to calculate
        self.ml         = {}
        self.std_fisher = {}
        self.cov_fisher = {"Iter 0":None }
        self.variables  = ''
        if fit=="alpha":
            ext_par = 0
            # add them later once you know how many are there
            self.ml["Iter 0"]         = {}
            self.std_fisher["Iter 0"] = {}
        elif fit=="Ad + alpha":
            ext_par = 1
            self.ml["Iter 0"]         = { 'Ad':1.0  } 
            self.std_fisher["Iter 0"] = { 'Ad':None }
            self.variables           += 'Ad'
        elif fit=="beta + alpha":
            ext_par = 1
            self.ml["Iter 0"]         = { 'beta':beta_ini  }
            self.std_fisher["Iter 0"] = { 'beta':None }
            self.variables           += 'beta'
        elif fit=="As + Ad + alpha":
            ext_par = 2
            self.ml["Iter 0"]         = { 'As':1.0,  'Ad':1.0  }
            self.std_fisher["Iter 0"] = { 'As':None, 'Ad':None }
            self.variables           += 'As, Ad'
        elif fit=="Ad + beta + alpha":
            ext_par = 2
            self.ml["Iter 0"]         = { 'Ad':1.0,  'beta':beta_ini}
            self.std_fisher["Iter 0"] = { 'Ad':None, 'beta':0.0 }
            self.variables           += 'Ad, beta'
        elif fit=="As + Ad + beta + alpha":
            ext_par = 3
            self.ml["Iter 0"]         = { 'As':1.0, 'Ad':1.0, 'beta':beta_ini}
            self.std_fisher["Iter 0"] = { 'As':None,'Ad':None,'beta':None }
            self.variables           += 'As, Ad, beta'
        elif fit=="As + Asd + Ad + alpha":
            ext_par = 3
            self.ml["Iter 0"]         = { 'As':1.0,  'Asd':1.0,  'Ad':1.0 }
            self.std_fisher["Iter 0"] = { 'As':None, 'Asd':None, 'Ad':None}
            self.variables           += 'As, Asd, Ad' 
        elif fit=="As + Asd + Ad + beta + alpha":
            ext_par = 4
            self.ml["Iter 0"]         = { 'As':1.0, 'Asd':1.0,  'Ad':1.0, 'beta':beta_ini}
            self.std_fisher["Iter 0"] = { 'As':None,'Asd':None, 'Ad':None,'beta':None }
            self.variables           += 'As, Asd, Ad, beta'
            
        if alpha_per_split:
            self.Nalpha = spec.Nbands
            for ii, band in enumerate(spec.bands):
                self.ml["Iter 0"][band]         = alpha_ini
                self.std_fisher["Iter 0"][band] = None
                self.variables                 += f'{band}'if (ii==0 and fit=="alpha") else f', {band}'
        else:
            self.Nalpha = spec.Nfreq
            for ii, freq in enumerate(spec.freqs):
                self.ml["Iter 0"][freq]         = alpha_ini
                self.std_fisher["Iter 0"][freq] = None
                self.variables                 += f'{freq}'if (ii==0 and fit=="alpha") else f', {freq}'

        self.ext_par = ext_par
        self.Nvar    = self.Nalpha + self.ext_par
        
      
class LinearSystem:
    mode_options = ['total', 'cumulative', 'ell']

    def __init__(self, mle, inv_cov, mode="total", window=5):
        self.dt             = np.float64
        self.fit            = mle.fit
        self.bin_cl         = mle.bin_terms
        self.iC             = inv_cov
        self.Nbands         = mle.Nbands
        self.mle            = mle
        
        assert mode in self.mode_options, f"mode must be one of {self.mode_options}"
        self.mode           = mode
        self.window         = window
        self.Nbins          = mle.Nbins
        if mode=="total":
            self.ext_dim = 1
        elif mode=="cumulative":
            self.ext_dim = self.Nbins
        elif mode=="ell":
            self.ext_dim = self.Nbins - (self.window - 1)

        # common to all fits (basic alpha*alpha fit)
        self.B_ijpq         = np.zeros((self.Nbands, self.Nbands, self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
        self.E_ijpq         = np.zeros((self.Nbands, self.Nbands, self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
        self.I_ijpq         = np.zeros((self.Nbands, self.Nbands, self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
        self.D_ij           = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
        self.H_ij           = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
        self.A              = np.zeros(self.ext_dim, dtype=self.dt)
        # only used in particular combinations
        if 'beta' in self.fit:
            # beta*beta and beta*alpha 
            self.tau_ij     = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
            self.varphi_ij  = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
            self.ene_ij     = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt) 
            self.epsilon_ij = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
            self.C          = np.zeros(self.ext_dim, dtype=self.dt) 
            self.F          = np.zeros(self.ext_dim, dtype=self.dt) 
            self.G          = np.zeros(self.ext_dim, dtype=self.dt) 
            self.O          = np.zeros(self.ext_dim, dtype=self.dt) 
            self.P          = np.zeros(self.ext_dim, dtype=self.dt)
        if 'Ad' in self.fit:
            # Ad*Ad and Ad*alpha
            self.sigma_ij   = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
            self.omega_ij   = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
            self.R          = np.zeros(self.ext_dim, dtype=self.dt)
            self.N          = np.zeros(self.ext_dim, dtype=self.dt)
            if 'beta' in self.fit:
                # Ad*beta
                self.LAMBDA = np.zeros(self.ext_dim, dtype=self.dt)
                self.mu     = np.zeros(self.ext_dim, dtype=self.dt)
        if 'As' in self.fit:
            # As*As and As*alpha
            self.nu_ij      = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
            self.psi_ij     = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt) 
            self.S          = np.zeros(self.ext_dim, dtype=self.dt)
            self.J          = np.zeros(self.ext_dim, dtype=self.dt)
            if 'beta' in self.fit:
                # As*beta
                self.X      = np.zeros(self.ext_dim, dtype=self.dt)
                self.Y      = np.zeros(self.ext_dim, dtype=self.dt)
            if 'Ad' in self.fit:
                # As*Ad
                self.W      = np.zeros(self.ext_dim, dtype=self.dt)
        if 'Asd' in self.fit:
            # Asd*Asd and Asd*alpha
            self.pi_ij      = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
            self.rho_ij     = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
            self.phi_ij     = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt) 
            self.OMEGA_ij   = np.zeros((self.Nbands, self.Nbands, self.ext_dim), dtype=self.dt)
            self.T          = np.zeros(self.ext_dim, dtype=self.dt)
            self.U          = np.zeros(self.ext_dim, dtype=self.dt)
            self.Z          = np.zeros(self.ext_dim, dtype=self.dt)
            self.M          = np.zeros(self.ext_dim, dtype=self.dt)
            self.L          = np.zeros(self.ext_dim, dtype=self.dt)
            if 'beta' in self.fit:
                # Asd*beta
                self.DELTA  = np.zeros(self.ext_dim, dtype=self.dt)
                self.eta    = np.zeros(self.ext_dim, dtype=self.dt) 
                self.theta  = np.zeros(self.ext_dim, dtype=self.dt) 
                self.delta  = np.zeros(self.ext_dim, dtype=self.dt)
            if 'Ad' in self.fit:
                # Asd*Ad
                self.K      = np.zeros(self.ext_dim, dtype=self.dt)
                self.xi     = np.zeros(self.ext_dim, dtype=self.dt)
            if 'As' in self.fit:
                # Asd*As
                self.Q      = np.zeros(self.ext_dim, dtype=self.dt)
                self.V      = np.zeros(self.ext_dim, dtype=self.dt)

    def compute_terms(self):
        if self.mode=="total":
            return self.__total__()
        elif self.mode=="cumulative":
            return self.__cumulative__() 
        elif self.mode=="ell":
            return self.__ell__()
          
    def __cumulative__(self):
        for MN_pair in self.mle.MNidx:
            ii, jj, pp, qq, mm, nn = self.mle.get_index(MN_pair)
            # common to all fits (basic alpha*alpha fit)
            self.B_ijpq[ii,jj,pp,qq,:]    = np.cumsum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBo_ij_b'][pp,qq,:])
            self.E_ijpq[ii,jj,pp,qq,:]    = np.cumsum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEo_ij_b'][pp,qq,:])
            self.I_ijpq[ii,jj,pp,qq,:]    = np.cumsum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEo_ij_b'][pp,qq,:]) 
            self.D_ij[ii,jj,:]           += np.cumsum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBo_ij_b'][pp,qq,:])
            self.H_ij[ii,jj,:]           += np.cumsum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBo_ij_b'][pp,qq,:])
            self.A[:]                    += np.cumsum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBo_ij_b'][pp,qq,:])
            # only used in particular combinations
            if 'beta' in self.fit:
                # beta*beta and beta*alpha 
                self.tau_ij[ii,jj,:]     += np.cumsum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:])
                self.varphi_ij[ii,jj,:]  += np.cumsum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
                self.ene_ij[ii,jj,:]     += np.cumsum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:])
                self.epsilon_ij[ii,jj,:] += np.cumsum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
                self.C[:]                += np.cumsum(self.bin_cl['EEcmb_ij_b'][ii,jj,:] *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:]) 
                self.F[:]                += np.cumsum(self.bin_cl['EEcmb_ij_b'][ii,jj,:] *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:]) 
                self.G[:]                += np.cumsum(self.bin_cl['BBcmb_ij_b'][ii,jj,:] *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:]) 
                self.O[:]                += np.cumsum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:]) 
                self.P[:]                += np.cumsum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
            if 'Ad' in self.fit:
                # Ad*Ad and Ad*alpha
                self.sigma_ij[ii,jj,:]   += np.cumsum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                self.omega_ij[ii,jj,:]   += np.cumsum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                self.R[:]                += np.cumsum(self.bin_cl['EBd_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                self.N[:]                += np.cumsum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                if 'beta' in self.fit:
                    # Ad*beta
                    self.LAMBDA[:]       += np.cumsum(self.bin_cl['EBd_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:])
                    self.mu[:]           += np.cumsum(self.bin_cl['EBd_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
            if 'As' in self.fit:
                # As*As and As*alpha
                self.nu_ij[ii,jj,:]      += np.cumsum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:])
                self.psi_ij[ii,jj,:]     += np.cumsum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:]) 
                self.S[:]                += np.cumsum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:])
                self.J[:]                += np.cumsum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:])
                if 'beta' in self.fit:
                    # As*beta
                    self.X[:]            += np.cumsum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:])
                    self.Y[:]            += np.cumsum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
                if 'Ad' in self.fit:
                    # As*Ad
                    self.W[:]            += np.cumsum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
            if 'Asd' in self.fit:
                # Asd*Asd and Asd*alpha
                self.pi_ij[ii,jj,:]      += np.cumsum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:])
                self.rho_ij[ii,jj,:]     += np.cumsum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
                self.phi_ij[ii,jj,:]     += np.cumsum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:])
                self.OMEGA_ij[ii,jj,:]   += np.cumsum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
                self.T[:]                += np.cumsum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:])
                self.U[:]                += np.cumsum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
                self.Z[:]                += np.cumsum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
                self.M[:]                += np.cumsum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
                self.L[:]                += np.cumsum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:])
                if 'beta' in self.fit:
                    # Asd*beta
                    self.DELTA[:]        += np.cumsum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:]) 
                    self.eta[:]          += np.cumsum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:]) 
                    self.theta[:]        += np.cumsum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:]) 
                    self.delta[:]        += np.cumsum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
                if 'Ad' in self.fit:
                    # Asd*Ad
                    self.K[:]            += np.cumsum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                    self.xi[:]           += np.cumsum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                if 'As' in self.fit:
                    # Asd*As
                    self.Q[:]            += np.cumsum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:])
                    self.V[:]            += np.cumsum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
   
    #TODO PDP: these ones can be further optimised but I'm leaving them like this
    # for now to debug the new code structure first
    def __total__(self):
        for MN_pair in self.mle.MNidx:
            ii, jj, pp, qq, mm, nn = self.mle.get_index(MN_pair)
            # common to all fits (basic alpha*alpha fit)
            self.B_ijpq[ii,jj,pp,qq]    = np.sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBo_ij_b'][pp,qq,:])
            self.E_ijpq[ii,jj,pp,qq]    = np.sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEo_ij_b'][pp,qq,:])
            self.I_ijpq[ii,jj,pp,qq]    = np.sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEo_ij_b'][pp,qq,:]) 
            self.D_ij[ii,jj]           += np.sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBo_ij_b'][pp,qq,:])
            self.H_ij[ii,jj]           += np.sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBo_ij_b'][pp,qq,:])
            self.A                     += np.sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBo_ij_b'][pp,qq,:])
            # only used in particular combinations 
            if 'beta' in self.fit:
                # beta*beta and beta*alpha 
                self.tau_ij[ii,jj]     += np.sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:])
                self.varphi_ij[ii,jj]  += np.sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
                self.ene_ij[ii,jj]     += np.sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:])
                self.epsilon_ij[ii,jj] += np.sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
                self.C                 += np.sum(self.bin_cl['EEcmb_ij_b'][ii,jj,:] *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:]) 
                self.F                 += np.sum(self.bin_cl['EEcmb_ij_b'][ii,jj,:] *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:]) 
                self.G                 += np.sum(self.bin_cl['BBcmb_ij_b'][ii,jj,:] *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:]) 
                self.O                 += np.sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:]) 
                self.P                 += np.sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
            if 'Ad' in self.fit:
                # Ad*Ad and Ad*alpha
                self.sigma_ij[ii,jj]   += np.sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                self.omega_ij[ii,jj]   += np.sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                self.R                 += np.sum(self.bin_cl['EBd_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                self.N                 += np.sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                if 'beta' in self.fit:
                    # Ad*beta
                    self.LAMBDA        += np.sum(self.bin_cl['EBd_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:])
                    self.mu            += np.sum(self.bin_cl['EBd_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
            if 'As' in self.fit:
                # As*As and As*alpha
                self.nu_ij[ii,jj]      += np.sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:])
                self.psi_ij[ii,jj]     += np.sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:]) 
                self.S                 += np.sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:])
                self.J                 += np.sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:])
                if 'beta' in self.fit:
                    # As*beta
                    self.X             += np.sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:])
                    self.Y             += np.sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
                if 'Ad' in self.fit:
                    # As*Ad
                    self.W             += np.sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
            if 'Asd' in self.fit:
                # Asd*Asd and Asd*alpha
                self.pi_ij[ii,jj]      += np.sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:])
                self.rho_ij[ii,jj]     += np.sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
                self.phi_ij[ii,jj]     += np.sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:])
                self.OMEGA_ij[ii,jj]   += np.sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
                self.T                 += np.sum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:])
                self.U                 += np.sum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
                self.Z                 += np.sum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
                self.M                 += np.sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])
                self.L                 += np.sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:])
                if 'beta' in self.fit:
                    # Asd*beta
                    self.DELTA         += np.sum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:]) 
                    self.eta           += np.sum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:]) 
                    self.theta         += np.sum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:]) 
                    self.delta         += np.sum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:])
                if 'Ad' in self.fit:
                    # Asd*Ad
                    self.K             += np.sum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                    self.xi            += np.sum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:])
                if 'As' in self.fit:
                    # Asd*As
                    self.Q             += np.sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:])
                    self.V             += np.sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:])

    def __ell__(self):
        for MN_pair in self.mle.MNidx:
            ii, jj, pp, qq, mm, nn = self.mle.get_index(MN_pair)
            # common to all fits (basic alpha*alpha fit)
            self.B_ijpq[ii,jj,pp,qq,:]    = moving_sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBo_ij_b'][pp,qq,:], self.window)
            self.E_ijpq[ii,jj,pp,qq,:]    = moving_sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEo_ij_b'][pp,qq,:], self.window)
            self.I_ijpq[ii,jj,pp,qq,:]    = moving_sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEo_ij_b'][pp,qq,:], self.window) 
            self.D_ij[ii,jj,:]           += moving_sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBo_ij_b'][pp,qq,:], self.window)
            self.H_ij[ii,jj,:]           += moving_sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBo_ij_b'][pp,qq,:], self.window)
            self.A[:]                    += moving_sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBo_ij_b'][pp,qq,:], self.window)
            # only used in particular combinations
            if 'beta' in self.fit:
                # beta*beta and beta*alpha 
                self.tau_ij[ii,jj,:]     += moving_sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:], self.window)
                self.varphi_ij[ii,jj,:]  += moving_sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:], self.window)
                self.ene_ij[ii,jj,:]     += moving_sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:], self.window)
                self.epsilon_ij[ii,jj,:] += moving_sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:], self.window)
                self.C[:]                += moving_sum(self.bin_cl['EEcmb_ij_b'][ii,jj,:] *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:], self.window) 
                self.F[:]                += moving_sum(self.bin_cl['EEcmb_ij_b'][ii,jj,:] *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:], self.window) 
                self.G[:]                += moving_sum(self.bin_cl['BBcmb_ij_b'][ii,jj,:] *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:], self.window) 
                self.O[:]                += moving_sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:], self.window) 
                self.P[:]                += moving_sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:], self.window)
            if 'Ad' in self.fit:
                # Ad*Ad and Ad*alpha
                self.sigma_ij[ii,jj,:]   += moving_sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:], self.window)
                self.omega_ij[ii,jj,:]   += moving_sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:], self.window)
                self.R[:]                += moving_sum(self.bin_cl['EBd_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:], self.window)
                self.N[:]                += moving_sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:], self.window)
                if 'beta' in self.fit:
                    # Ad*beta
                    self.LAMBDA[:]       += moving_sum(self.bin_cl['EBd_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:], self.window)
                    self.mu[:]           += moving_sum(self.bin_cl['EBd_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:], self.window)
            if 'As' in self.fit:
                # As*As and As*alpha
                self.nu_ij[ii,jj,:]      += moving_sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:], self.window)
                self.psi_ij[ii,jj,:]     += moving_sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:], self.window) 
                self.S[:]                += moving_sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:], self.window)
                self.J[:]                += moving_sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBs_ij_b'][pp,qq,:], self.window)
                if 'beta' in self.fit:
                    # As*beta
                    self.X[:]            += moving_sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:], self.window)
                    self.Y[:]            += moving_sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:], self.window)
                if 'Ad' in self.fit:
                    # As*Ad
                    self.W[:]            += moving_sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:], self.window)
            if 'Asd' in self.fit:
                # Asd*Asd and Asd*alpha
                self.pi_ij[ii,jj,:]      += moving_sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:], self.window)
                self.rho_ij[ii,jj,:]     += moving_sum(self.bin_cl['EEo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:], self.window)
                self.phi_ij[ii,jj,:]     += moving_sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:], self.window)
                self.OMEGA_ij[ii,jj,:]   += moving_sum(self.bin_cl['BBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:], self.window)
                self.T[:]                += moving_sum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:], self.window)
                self.U[:]                += moving_sum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:], self.window)
                self.Z[:]                += moving_sum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:], self.window)
                self.M[:]                += moving_sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:], self.window)
                self.L[:]                += moving_sum(self.bin_cl['EBo_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:], self.window)
                if 'beta' in self.fit:
                    # Asd*beta
                    self.DELTA[:]        += moving_sum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:], self.window) 
                    self.eta[:]          += moving_sum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EEcmb_ij_b'][pp,qq,:], self.window) 
                    self.theta[:]        += moving_sum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:], self.window) 
                    self.delta[:]        += moving_sum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['BBcmb_ij_b'][pp,qq,:], self.window)
                if 'Ad' in self.fit:
                    # Asd*Ad
                    self.K[:]            += moving_sum(self.bin_cl['EdBs_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:], self.window)
                    self.xi[:]           += moving_sum(self.bin_cl['EsBd_ij_b'][ii,jj,:]  *self.iC[:,mm,nn]* self.bin_cl['EBd_ij_b'][pp,qq,:], self.window)
                if 'As' in self.fit:
                    # Asd*As
                    self.Q[:]            += moving_sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EsBd_ij_b'][pp,qq,:], self.window)
                    self.V[:]            += moving_sum(self.bin_cl['EBs_ij_b'][ii,jj,:]   *self.iC[:,mm,nn]* self.bin_cl['EdBs_ij_b'][pp,qq,:], self.window)



########################################################################       

class MLE:
    fit_options = ["alpha", "Ad + alpha", "As + Ad + alpha", "As + Asd + Ad + alpha",
                   "beta + alpha", "Ad + beta + alpha", "As + Ad + beta + alpha","As + Asd + Ad + beta + alpha"]
    
    def __init__(self, libdir, spec_lib, fit,
                 alpha_per_split=False,
                 rm_same_tube=False,
                 binwidth=20, bmin=51, bmax=1000):
        self.niter_max = 100
        self.tol       = 0.5 # arcmin  
        self.spec      = spec_lib
        self.libdir    = self.spec.lat.libdir+'/mle'
        os.makedirs(self.libdir, exist_ok=True)
        self.nside     = self.spec.nside
        self.cmb       = CMB(libdir, self.nside, beta=0, model='iso')
        self.cmb_cls   = self.cmb.get_lensed_spectra(dl=False, dtype='d')
        self.fsky      = self.spec.fsky
        
        # define binning
        assert bmax <= self.spec.lmax, "bmax must be less than lmax in Spectra object"
        self.nlb      = binwidth
        self.bmin     = bmin
        self.bmax     = bmax
        lower_edge    = np.arange(self.bmin,          self.bmax-self.nlb, self.nlb)
        upper_edge    = np.arange(self.bmin+self.nlb, self.bmax,          self.nlb)
        bin_def       = bin_from_edges(lower_edge, upper_edge)
        self.bin_conf = bin_configuration(bin_def)
        self.Nbins    = bin_def[0]

        # define instrument
        #TODO you could ask to use a specific combination of bands (excluding some)
        self.bands  = self.spec.bands
        self.Nbands = self.spec.Nbands
        self.inst   = {}
        for ii, band in enumerate(self.bands):
            self.inst[band] = {"fwhm": self.spec.lat.config[band]['fwhm'], 
                               "opt. tube": self.spec.lat.config[band]['opt. tube'], 
                               "cl idx":ii}
            
        # parameters to calculate
        assert fit in self.fit_options, f"fit must be one of {self.fit_options}"
        self.fit    = fit
        self.alpha_per_split = alpha_per_split
        if alpha_per_split:
            print("Fitting a different polarisation angle per split")
            for ii, band in enumerate(self.bands):
                self.inst[band]["alpha idx"]              = ii
        else:
            print("Fitting a common polarisation angle per frequency")
            counter = 0
            for ii, freq in enumerate(self.spec.freqs):
                for split in range(self.spec.lat.nsplits):
                     self.inst[f'{freq}-{split+1}']["alpha idx"] = counter
                counter += 1
        
        self.rm_same_tube = rm_same_tube
        if self.rm_same_tube:
            print("Don't use cross-spectra of bands within the same optical tube")
            avoid = 4 # remove auto-spectra and the 3 correlations between bands in the same tube
        else:
            avoid = 1 # always remove auto-spectra
        self.avoid = avoid

        # matrices for indexing
        self.MNi  = np.zeros((self.Nbands*(self.Nbands-avoid), self.Nbands*(self.Nbands-avoid)), dtype=np.uint8)
        self.MNj  = np.zeros((self.Nbands*(self.Nbands-avoid), self.Nbands*(self.Nbands-avoid)), dtype=np.uint8)
        self.MNp  = np.zeros((self.Nbands*(self.Nbands-avoid), self.Nbands*(self.Nbands-avoid)), dtype=np.uint8)
        self.MNq  = np.zeros((self.Nbands*(self.Nbands-avoid), self.Nbands*(self.Nbands-avoid)), dtype=np.uint8)
        
        IJidx = []
        for ii, band_i in enumerate(self.bands):
            for jj, band_j in enumerate(self.bands):
                if self.rm_same_tube:
                    if not self.same_tube(band_i, band_j):
                        IJidx.append((ii, jj))
                else:
                    if jj!=ii: 
                        IJidx.append((ii,jj))
        self.IJidx = np.array(IJidx, dtype=np.uint8)

        MNidx = [] 
        for mm in range(0, self.Nbands*(self.Nbands-avoid), 1):
            for nn in range(0, self.Nbands*(self.Nbands-avoid), 1):
                    MNidx.append((mm,nn))
        self.MNidx = np.array(MNidx, dtype=np.uint16) # data type valid for <=70 bands, optimizing memory use

        for MN_pair in self.MNidx:
            ii, jj, pp, qq, mm, nn =self.get_index(MN_pair)
            self.MNi[mm, nn] = ii; self.MNj[mm, nn] = jj
            self.MNp[mm, nn] = pp; self.MNq[mm, nn] = qq
            
            
            
    def same_tube(self, band_1, band_2):
        return self.inst[band_1]["opt. tube"] == self.inst[band_2]["opt. tube"]
    
    def get_index(self, mn_pair):
        mm, nn = mn_pair
        ii, jj = self.IJidx[mm]
        pp, qq = self.IJidx[nn]
        return ii, jj, pp, qq, mm, nn

    def convolve_gaussBeams_pwf(self, mode, fwhm1, fwhm2, lmax):
        assert mode in ["ee", "bb"], "mode must be 'ee' or 'bb'"
        (_, pwf) = hp.pixwin(self.nside, pol=True, lmax=lmax)
        bl_1 = hp.gauss_beam(fwhm1/rad2arcmin, lmax=lmax, pol=True)
        bl_2 = hp.gauss_beam(fwhm2/rad2arcmin, lmax=lmax, pol=True)
        if mode=='ee':
            bl = bl_1[:,1]*bl_2[:,1]
        elif mode=='bb':
            bl = bl_1[:,2]*bl_2[:,2]
        #pixel window function is squared because both are polarization fields
        return self.cmb_cls[mode][:lmax+1]*bl*pwf**2

    def __get_alpha_blocks__(self, Niter, res):
        alphas = np.zeros(self.Nbands, dtype=np.float64)
        for band in self.bands:
            alphas[self.inst[band]['cl idx']] = res.ml[f"Iter {Niter}"][band if self.alpha_per_split else band[:-2]]
        return alphas[self.MNi], alphas[self.MNj], alphas[self.MNp], alphas[self.MNq]

    def __get_ml_alphas__(self, Niter, res, add_beta=False):
        alphas = np.zeros(res.Nalpha, dtype=np.float64)
        if self.alpha_per_split:
            for ii, band in enumerate(self.bands):
                alphas[ii] = res.ml[f"Iter {Niter}"][band]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                alphas[ii] = res.ml[f"Iter {Niter}"][freq]
                
        if add_beta:
            alphas += res.ml[f"Iter {Niter}"]['beta']
        return alphas

############################################################################### 
### Combination of covariance matrix elements

    def build_cov(self, Niter, res):
        # get parameters for this iteration
        ai, aj, ap, aq = self.__get_alpha_blocks__(Niter, res)  
        # trigonometric factors rotating the spectra
        cicj = np.cos(2*ai)*np.cos(2*aj); cpcq = np.cos(2*ap)*np.cos(2*aq)
        sisj = np.sin(2*ai)*np.sin(2*aj); spsq = np.sin(2*ap)*np.sin(2*aq)
        c4ij = np.cos(4*ai)+np.cos(4*aj); c4pq = np.cos(4*ap)+np.cos(4*aq)
        Aij  = np.sin(4*aj)/c4ij;         Apq  = np.sin(4*aq)/c4pq
        Bij  = np.sin(4*ai)/c4ij;         Bpq  = np.sin(4*ap)/c4pq   
        Dij  = 2*cicj/c4ij      ;         Dpq  = 2*cpcq/c4pq
        Eij  = 2*sisj/c4ij      ;         Epq  = 2*spsq/c4pq  
        
        # observed * observed; remove all EB except the one in T0
        To   = np.copy(self.cov_terms['C_oxo'])
        cov  = To[0,:,:,:] + Apq*Aij*To[1,:,:,:] + Bpq*Bij*To[2,:,:,:]
        
        if "beta" in self.fit:
            beta = res.ml[f"Iter {Niter}"]["beta"]
            Cij  = np.sin(4*beta)/(2*np.cos(2*ai+2*aj))
            Cpq  = np.sin(4*beta)/(2*np.cos(2*ap+2*aq))
            Tcmb = np.copy(self.cov_terms['C_cmb'])
            # cmb * cmb + cmb * observed
            cov += - 2*Cij*Cpq*( Tcmb[0,:,:,:] + Tcmb[1,:,:,:] )
            
        if "Ad" in self.fit:
            Ad   = res.ml[f"Iter {Niter}"]["Ad"]
            Td   = np.copy(self.cov_terms['C_dxd'])
            Td_o = np.copy(self.cov_terms['C_dxo'])
            # dust * dust; remove EB from T1, T2, T3
            cov += + Dij*Dpq*Ad**2*Td[0,:,:,:] + Eij*Epq*Ad**2*Td[1,:,:,:] + Dij*Epq*Ad**2*Td[2,:,:,:] + Eij*Dpq*Ad**2*Td[3,:,:,:]
            # dust * observed; remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
            cov += - Dij*Ad*Td_o[0,:,:,:] - Dpq*Ad*Td_o[1,:,:,:] - Eij*Ad*Td_o[2,:,:,:] - Epq*Ad*Td_o[3,:,:,:]
        
        if "As" in self.fit:
            As   = res.ml[f"Iter {Niter}"]["As"]
            Ts   = np.copy(self.cov_terms['C_sxs'])
            Ts_d = np.copy(self.cov_terms['C_sxd'])
            Ts_o = np.copy(self.cov_terms['C_sxo'])
            # synch * synch; remove EB from T1, T2, T3
            cov += + Dij*Dpq*As**2*Ts[0,:,:,:] + Eij*Epq*As**2*Ts[1,:,:,:] + Dij*Epq*As**2*Ts[2,:,:,:] + Eij*Dpq*As**2*Ts[3,:,:,:]
            # synch * observed; remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
            cov += - Dij*As*Ts_o[0,:,:,:] - Dpq*As*Ts_o[1,:,:,:]  - Eij*As*Ts_o[2,:,:,:] - Epq*As*Ts_o[3,:,:,:]
            # synch * dust; remove EB from T2, T3, T4, T5, T6, T7
            cov += + Dij*Dpq*As*Ad*( Ts_d[0,:,:,:] + Ts_d[1,:,:,:] ) + Eij*Epq*As*Ad*( Ts_d[2,:,:,:] + Ts_d[3,:,:,:] ) 
            cov += + Dij*Epq*As*Ad*( Ts_d[4,:,:,:] + Ts_d[7,:,:,:] ) + Dpq*Eij*As*Ad*( Ts_d[5,:,:,:] + Ts_d[6,:,:,:] )
        
        if "Asd" in self.fit:
            Asd    = res.ml[f"Iter {Niter}"]["Asd"]
            TSD    = np.copy(self.cov_terms['C_sdxsd'])
            TDS    = np.copy(self.cov_terms['C_dsxds'])
            TSD_DS = np.copy(self.cov_terms['C_sdxds'])
            Ts_SD  = np.copy(self.cov_terms['C_sxsd'])
            Ts_DS  = np.copy(self.cov_terms['C_sxds'])
            Td_SD  = np.copy(self.cov_terms['C_dxsd'])
            Td_DS  = np.copy(self.cov_terms['C_dxds'])
            TSD_o  = np.copy(self.cov_terms['C_sdxo'])
            TDS_o  = np.copy(self.cov_terms['C_dsxo'])
            # covariance elements
            # synch-dust * synch-dust; remove EB from T1, T2, T3
            cov += + Dij*Dpq*Asd**2*TSD[0,:,:,:] + Eij*Epq*Asd**2*TSD[1,:,:,:] + Dij*Epq*Asd**2*TSD[2,:,:,:] + Eij*Dpq*Asd**2*TSD[3,:,:,:]
            # dust-synch * dust-synch; remove EB from T1, T2, T3
            cov += + Dij*Dpq*Asd**2*TDS[0,:,:,:] + Eij*Epq*Asd**2*TDS[1,:,:,:] + Dij*Epq*Asd**2*TDS[2,:,:,:] + Eij*Dpq*Asd**2*TDS[3,:,:,:]
            ## synch-dust * dust-synch; remove EB from T2, T3, T4, T5, T6, T7
            cov += + Dij*Dpq*Asd**2*( TSD_DS[0,:,:,:] + TSD_DS[1,:,:,:] ) + Eij*Epq*Asd**2*( TSD_DS[2,:,:,:] + TSD_DS[3,:,:,:] )
            cov += + Dij*Epq*Asd**2*( TSD_DS[4,:,:,:] + TSD_DS[7,:,:,:] ) + Dpq*Eij*Asd**2*( TSD_DS[5,:,:,:] + TSD_DS[6,:,:,:] )
            # synch * synch-dust; remove EB from T2, T3, T4, T5, T6, T7
            cov += + Dij*Dpq*As*Asd*( Ts_SD[0,:,:,:] + Ts_SD[1,:,:,:] ) + Eij*Epq*As*Asd*( Ts_SD[6,:,:,:] + Ts_SD[7,:,:,:] )
            cov += + Dij*Epq*As*Asd*( Ts_SD[2,:,:,:] + Ts_SD[5,:,:,:] ) + Dpq*Eij*As*Asd*( Ts_SD[3,:,:,:] + Ts_SD[4,:,:,:] )
            # synch * dust-synch; remove EB from T2, T3, T4, T5, T6, T7
            cov += + Dij*Dpq*As*Asd*( Ts_DS[0,:,:,:] + Ts_DS[1,:,:,:] ) + Eij*Epq*As*Asd*( Ts_DS[6,:,:,:] + Ts_DS[7,:,:,:] )
            cov += + Dij*Epq*As*Asd*( Ts_DS[2,:,:,:] + Ts_DS[5,:,:,:] ) + Dpq*Eij*As*Asd*( Ts_DS[3,:,:,:] + Ts_DS[4,:,:,:] )
            # dust * synch-dust; remove EB from T2, T3, T4, T5, T6, T7
            cov += + Dij*Dpq*Ad*Asd*( Td_SD[0,:,:,:] + Td_SD[1,:,:,:] ) + Eij*Epq*Ad*Asd*( Td_SD[6,:,:,:] + Td_SD[7,:,:,:] )
            cov += + Dij*Epq*Ad*Asd*( Td_SD[2,:,:,:] + Td_SD[5,:,:,:] ) + Dpq*Eij*Ad*Asd*( Td_SD[3,:,:,:] + Td_SD[4,:,:,:] )
            # dust * dust-synch; remove EB from T2, T3, T4, T5, T6, T7
            cov += + Dij*Dpq*Ad*Asd*( Td_DS[0,:,:,:] + Td_DS[1,:,:,:] ) + Eij*Epq*Ad*Asd*( Td_DS[6,:,:,:] + Td_DS[7,:,:,:] )
            cov += + Dij*Epq*Ad*Asd*( Td_DS[2,:,:,:] + Td_DS[5,:,:,:] ) + Dpq*Eij*Ad*Asd*( Td_DS[3,:,:,:] + Td_DS[4,:,:,:] )
            # synch-dust * observed; remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
            cov += - Dij*Asd*TSD_o[0,:,:,:] - Dpq*Asd*TSD_o[1,:,:,:] - Eij*Asd*TSD_o[2,:,:,:] - Epq*Asd*TSD_o[3,:,:,:]
            # dust-synch * observed; remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
            cov += - Dij*Asd*TDS_o[0,:,:,:] - Dpq*Asd*TDS_o[1,:,:,:] - Eij*Asd*TDS_o[2,:,:,:] - Epq*Asd*TDS_o[3,:,:,:]
            
        return cov
               
    
###############################################################################
### Calculation of covariance matrix elements

    def C_cmb(self):  
        lmax  = self.spec.lmax
        bl_EE = np.zeros((self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        bl_BB = np.zeros((self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        for MN_pair in self.MNidx:
            ii, jj, pp ,qq, mm, nn = self.get_index(MN_pair)
            
            b_i = hp.gauss_beam(self.inst[self.bands[ii]]['fwhm']/rad2arcmin, lmax=lmax, pol=True)
            b_j = hp.gauss_beam(self.inst[self.bands[jj]]['fwhm']/rad2arcmin, lmax=lmax, pol=True)
            b_p = hp.gauss_beam(self.inst[self.bands[pp]]['fwhm']/rad2arcmin, lmax=lmax, pol=True) 
            b_q = hp.gauss_beam(self.inst[self.bands[qq]]['fwhm']/rad2arcmin, lmax=lmax, pol=True) 
            
            bl_EE[mm, nn, :] = b_i[:,1]*b_j[:,1]*b_p[:,1]*b_q[:,1]  
            bl_BB[mm, nn, :] = b_i[:,2]*b_j[:,2]*b_p[:,2]*b_q[:,2] 

        ell      = np.arange(0, lmax+1, 1)
        (_, pwf) = hp.pixwin(self.nside, pol=True, lmax=lmax)
        ##################### cmb 
        Tcmb = np.zeros((2, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        Tcmb[0,:,:,:] = pwf**4 * bl_EE * self.cmb_cls['ee'][:lmax+1]**2 /(2*ell+1)
        Tcmb[1,:,:,:] = pwf**4 * bl_BB * self.cmb_cls['bb'][:lmax+1]**2 /(2*ell+1)     
        return np.moveaxis(bin_cov_matrix(Tcmb, self.bin_conf), 3, 1)

    def C_oxo(self, EiEjo, BiBjo, EiBjo, BiEjo):  
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1, 1)
        ############################ remove all except T0(1), T5(6), T6(7)
        # observed * observed
        To = np.zeros((3, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        #(1) observed
        To[0,:,:,:] = (EiEjo[self.MNi,self.MNp,:]*BiBjo[self.MNj,self.MNq,:] + EiBjo[self.MNi,self.MNq,:]*BiEjo[self.MNj,self.MNp,:])/(2*ell+1)
        #(6) observed
        To[1,:,:,:] = (EiEjo[self.MNi,self.MNp,:]*EiEjo[self.MNj,self.MNq,:] + EiEjo[self.MNi,self.MNq,:]*EiEjo[self.MNj,self.MNp,:])/(2*ell+1)
        #(7) observed
        To[2,:,:,:] = (BiBjo[self.MNi,self.MNp,:]*BiBjo[self.MNj,self.MNq,:] + BiBjo[self.MNi,self.MNq,:]*BiBjo[self.MNj,self.MNp,:])/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(To, self.bin_conf), 3, 1)
        
    def C_fgxfg(self, EiEj, BiBj, EiBj, BiEj):  
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1, 1)
        ##################### remove EB from T1, T2, T3
        # synch * synch or dust * dust
        Tfg = np.zeros((4, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        #(1s)
        Tfg[0,:,:,:] = (EiEj[self.MNi,self.MNp,:]*BiBj[self.MNj,self.MNq,:] + EiBj[self.MNi,self.MNq,:]*BiEj[self.MNj,self.MNp,:])/(2*ell+1)
        #(2s)
        Tfg[1,:,:,:] = BiBj[self.MNi,self.MNp,:]*EiEj[self.MNj,self.MNq,:]/(2*ell+1) 
        #(3s)
        Tfg[2,:,:,:] = EiEj[self.MNi,self.MNq,:]*BiBj[self.MNj,self.MNp,:]/(2*ell+1)
        #(4s)
        Tfg[3,:,:,:] = BiBj[self.MNi,self.MNq,:]*EiEj[self.MNj,self.MNp,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(Tfg, self.bin_conf), 3, 1)
        
    def C_sdxsd(self, EiEjs, BiBjs, EiBjs, BiEjs, EiEjd, BiBjd, EiBjd, BiEjd,
                Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd):  
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1, 1)
        ##################### remove EB from T1, T2, T3
        # synch-dust * synch-dust 
        TSD = np.zeros((4, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1SD)
        TSD[0,:,:,:] = (EiEjs[self.MNi,self.MNp,:]*BiBjd[self.MNj,self.MNq,:] + Eis_Bjd[self.MNi,self.MNq,:]*Eis_Bjd[self.MNp,self.MNj,:])/(2*ell+1)
        # (2SD)
        TSD[1,:,:,:] = BiBjs[self.MNi,self.MNp,:]*EiEjd[self.MNj,self.MNq,:]/(2*ell+1) 
        # (3SD)
        TSD[2,:,:,:] = Eis_Ejd[self.MNi,self.MNq,:]*Bis_Bjd[self.MNp,self.MNj,:]/(2*ell+1)
        # (4SD)
        TSD[3,:,:,:] = Bis_Bjd[self.MNi,self.MNq,:]*Eis_Ejd[self.MNp,self.MNj,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(TSD, self.bin_conf), 3, 1)
    
    def C_dsxds(self, EiEjs, BiBjs, EiBjs, BiEjs, EiEjd, BiBjd, EiBjd, BiEjd,
                Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd):  
            
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1, 1)   
        ##################### remove EB from T1, T2, T3
        # dust-synch * dust-synch 
        TDS = np.zeros((4, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1DS)
        TDS[0,:,:,:] = (EiEjd[self.MNi,self.MNp,:]*BiBjs[self.MNj,self.MNq,:] + Bis_Ejd[self.MNq,self.MNi,:]*Bis_Ejd[self.MNj,self.MNp,:])/(2*ell+1)
        # (2DS)
        TDS[1,:,:,:] = BiBjd[self.MNi,self.MNp,:]*EiEjs[self.MNj,self.MNq,:]/(2*ell+1) 
        # (3DS)
        TDS[2,:,:,:] = Eis_Ejd[self.MNq,self.MNi,:]*Bis_Bjd[self.MNj,self.MNp,:]/(2*ell+1)
        # (4DS)
        TDS[3,:,:,:] = Bis_Bjd[self.MNq,self.MNi,:]*Eis_Ejd[self.MNj,self.MNp,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(TDS, self.bin_conf), 3, 1)

    def C_fgxo(self,Eifg_Ejo, Bifg_Bjo, Eifg_Bjo, Bifg_Ejo):  
        lmax = self.spec.lmax
        ell=np.arange(0, lmax+1, 1)
        ##################### remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        # synch * observed  or dust * obs
        Tfg_o = np.zeros((4,self.Nbands*(self.Nbands-self.avoid),self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1so)
        Tfg_o[0,:,:,:] = (Eifg_Ejo[self.MNi,self.MNp,:]*Bifg_Bjo[self.MNj,self.MNq,:] + Eifg_Bjo[self.MNi,self.MNq,:]*Bifg_Ejo[self.MNj,self.MNp,:])/(2*ell+1)
        # (1so*)
        Tfg_o[1,:,:,:] = (Eifg_Ejo[self.MNp,self.MNi,:]*Bifg_Bjo[self.MNq,self.MNj,:] + Eifg_Bjo[self.MNp,self.MNj,:]*Bifg_Ejo[self.MNq,self.MNi,:])/(2*ell+1)
        # (4so)
        Tfg_o[2,:,:,:] =  Bifg_Bjo[self.MNi,self.MNq,:]*Eifg_Ejo[self.MNj,self.MNp,:]/(2*ell+1) 
        # (4so*)
        Tfg_o[3,:,:,:] =  Bifg_Bjo[self.MNp,self.MNj,:]*Eifg_Ejo[self.MNq,self.MNi,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(Tfg_o, self.bin_conf), 3, 1)
        
    def C_sdxo(self,Eis_Ejo, Bis_Bjo, Eis_Bjo, Bis_Ejo, Eid_Ejo, Bid_Bjo, Eid_Bjo, Bid_Ejo):  
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1, 1)
        ##################### remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        # synch-dust * observed 
        TSD_o = np.zeros((4,self.Nbands*(self.Nbands-self.avoid),self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1SDo)
        TSD_o[0,:,:,:] = (Eis_Ejo[self.MNi,self.MNp,:]*Bid_Bjo[self.MNj,self.MNq,:] + Eis_Bjo[self.MNi,self.MNq,:]*Bid_Ejo[self.MNj,self.MNp,:])/(2*ell+1)
        # (1SDo*)
        TSD_o[1,:,:,:] = (Eis_Ejo[self.MNp,self.MNi,:]*Bid_Bjo[self.MNq,self.MNj,:] + Eis_Bjo[self.MNp,self.MNj,:]*Bid_Ejo[self.MNq,self.MNi,:])/(2*ell+1)
        # (4SDo)
        TSD_o[2,:,:,:] = Bis_Bjo[self.MNi,self.MNq,:]*Eid_Ejo[self.MNj,self.MNp,:]/(2*ell+1)
        # (4SDo*)
        TSD_o[3,:,:,:] = Bis_Bjo[self.MNp,self.MNj,:]*Eid_Ejo[self.MNq,self.MNi,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(TSD_o, self.bin_conf), 3, 1)
    
    def C_dsxo(self,Eis_Ejo, Bis_Bjo, Eis_Bjo, Bis_Ejo, Eid_Ejo, Bid_Bjo, Eid_Bjo, Bid_Ejo):  
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1, 1)
        ##################### remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        # dust-synch * observed  
        TDS_o = np.zeros((4, self.Nbands*(self.Nbands-self.avoid),self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1DSo)
        TDS_o[0,:,:,:] = (Eid_Ejo[self.MNi,self.MNp,:]*Bis_Bjo[self.MNj,self.MNq,:] + Eid_Bjo[self.MNi,self.MNq,:]*Bis_Ejo[self.MNj,self.MNp,:])/(2*ell+1)
        # (1DSo*)
        TDS_o[1,:,:,:] = (Eid_Ejo[self.MNp,self.MNi,:]*Bis_Bjo[self.MNq,self.MNj,:] + Eid_Bjo[self.MNp,self.MNj,:]*Bis_Ejo[self.MNq,self.MNi,:])/(2*ell+1)
        # (4DSo)
        TDS_o[2,:,:,:] = Bid_Bjo[self.MNi,self.MNq,:]*Eis_Ejo[self.MNj,self.MNp,:]/(2*ell+1)
        # (4DSo*)
        TDS_o[3,:,:,:] = Bid_Bjo[self.MNp,self.MNj,:]*Eis_Ejo[self.MNq,self.MNi,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(TDS_o, self.bin_conf), 3, 1)

    def C_sxd(self, Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd):  
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1, 1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # synch * dust 
        Ts_d=np.zeros((8, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1sd)
        Ts_d[0,:,:,:] = (Eis_Ejd[self.MNi,self.MNp,:]*Bis_Bjd[self.MNj,self.MNq,:] + Eis_Bjd[self.MNi,self.MNq,:]*Bis_Ejd[self.MNj,self.MNp,:])/(2*ell+1)
        # (1sd*)
        Ts_d[1,:,:,:] = (Eis_Ejd[self.MNp,self.MNi,:]*Bis_Bjd[self.MNq,self.MNj,:] + Eis_Bjd[self.MNp,self.MNj,:]*Bis_Ejd[self.MNq,self.MNi,:])/(2*ell+1)
        # (2sd)
        Ts_d[2,:,:,:] = Bis_Bjd[self.MNi,self.MNp,:]*Eis_Ejd[self.MNj,self.MNq,:]/(2*ell+1)
        # (2sd*)
        Ts_d[3,:,:,:] = Bis_Bjd[self.MNp,self.MNi,:]*Eis_Ejd[self.MNq,self.MNj,:]/(2*ell+1)
        # (3sd) corregido
        Ts_d[4,:,:,:] = Eis_Ejd[self.MNi,self.MNq,:]*Bis_Bjd[self.MNj,self.MNp,:]/(2*ell+1)
        # (3sd*) corregido
        Ts_d[5,:,:,:] = Eis_Ejd[self.MNp,self.MNj,:]*Bis_Bjd[self.MNq,self.MNi,:]/(2*ell+1)
        # (4sd) corregido
        Ts_d[6,:,:,:] = Bis_Bjd[self.MNi,self.MNq,:]*Eis_Ejd[self.MNj,self.MNp,:]/(2*ell+1)
        # (4sd*) corregido
        Ts_d[7,:,:,:] = Bis_Bjd[self.MNp,self.MNj,:]*Eis_Ejd[self.MNq,self.MNi,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(Ts_d, self.bin_conf), 3, 1)
    
    def C_sdxds(self, EiEjs, BiBjs, EiBjs, BiEjs, EiEjd, BiBjd, EiBjd, BiEjd,
                Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd):  
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1,1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # synch-dust * dust-synch
        TSD_DS = np.zeros((8, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1SDDS)
        TSD_DS[0,:,:,:] = (Eis_Ejd[self.MNi,self.MNp,:]*Bis_Bjd[self.MNq,self.MNj,:] + EiBjs[self.MNi,self.MNq,:]*BiEjd[self.MNj,self.MNp,:])/(2*ell+1)
        # (1SDDS*)
        TSD_DS[1,:,:,:] = (Eis_Ejd[self.MNp,self.MNi,:]*Bis_Bjd[self.MNj,self.MNq,:] + EiBjd[self.MNi,self.MNq,:]*BiEjs[self.MNj,self.MNp,:])/(2*ell+1)
        # (2SDDS)
        TSD_DS[2,:,:,:] = Bis_Bjd[self.MNi,self.MNp,:]*Eis_Ejd[self.MNq,self.MNj,:]/(2*ell+1)
        # (2SDDS*)
        TSD_DS[3,:,:,:] = Bis_Bjd[self.MNp,self.MNi,:]*Eis_Ejd[self.MNj,self.MNq,:]/(2*ell+1)
        # (3SDDS)
        TSD_DS[4,:,:,:] = EiEjs[self.MNi,self.MNq,:]*BiBjd[self.MNj,self.MNp,:]/(2*ell+1)
        # (3SDDS*)
        TSD_DS[5,:,:,:] = BiBjd[self.MNi,self.MNq,:]*EiEjs[self.MNj,self.MNp,:]/(2*ell+1)
        # (4SDDS)
        TSD_DS[6,:,:,:] = BiBjs[self.MNi,self.MNq,:]*EiEjd[self.MNj,self.MNp,:]/(2*ell+1)
        # (4SDDS*)
        TSD_DS[7,:,:,:] = EiEjd[self.MNi,self.MNq,:]*BiBjs[self.MNj,self.MNp,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(TSD_DS, self.bin_conf), 3, 1)
        
#TODO terms have a different order but if you are careful you could build C_fgxsd and C_fgxds    
    def C_sxsd(self,EiEjs, BiBjs, EiBjs, BiEjs, Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd):  
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1, 1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # synch * synch-dust 
        Ts_SD = np.zeros((8, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1sSD)
        Ts_SD[0,:,:,:] = (EiEjs[self.MNi,self.MNp,:]*Bis_Bjd[self.MNj,self.MNq,:] + Eis_Bjd[self.MNi,self.MNq,:]*BiEjs[self.MNj,self.MNp,:])/(2*ell+1)
        # (1sSD*)
        Ts_SD[1,:,:,:] = (EiEjs[self.MNp,self.MNi,:]*Bis_Bjd[self.MNq,self.MNj,:] + Eis_Bjd[self.MNp,self.MNj,:]*BiEjs[self.MNq,self.MNi,:])/(2*ell+1)
        # (2sSD)
        Ts_SD[2,:,:,:] = Eis_Ejd[self.MNi,self.MNq,:]*BiBjs[self.MNj,self.MNp,:]/(2*ell+1)
        # (2sSD*)
        Ts_SD[3,:,:,:] = Eis_Ejd[self.MNp,self.MNj,:]*BiBjs[self.MNq,self.MNi,:]/(2*ell+1)
        # (3sSD)
        Ts_SD[4,:,:,:] = Bis_Bjd[self.MNi,self.MNq,:]*EiEjs[self.MNj,self.MNp,:]/(2*ell+1)
        # (3sSD*)
        Ts_SD[5,:,:,:] = Bis_Bjd[self.MNp,self.MNj,:]*EiEjs[self.MNq,self.MNi,:]/(2*ell+1)
        # (4sSD)
        Ts_SD[6,:,:,:] = BiBjs[self.MNi,self.MNp,:]*Eis_Ejd[self.MNj,self.MNq,:]/(2*ell+1)
        # (4sSD*)
        Ts_SD[7,:,:,:] = BiBjs[self.MNp,self.MNi,:]*Eis_Ejd[self.MNq,self.MNj,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(Ts_SD, self.bin_conf), 3, 1)
        
    def C_sxds(self,EiEjs, BiBjs, EiBjs, BiEjs, Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd):  
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1, 1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # synch * dust-synch 
        Ts_DS=np.zeros((8, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1sDS)
        Ts_DS[0,:,:,:] = (Eis_Ejd[self.MNi,self.MNp,:]*BiBjs[self.MNj,self.MNq,:] + EiBjs[self.MNi,self.MNq,:]*Bis_Ejd[self.MNj,self.MNp,:])/(2*ell+1)
        # (1sDS*)
        Ts_DS[1,:,:,:] = (Eis_Ejd[self.MNp,self.MNi,:]*BiBjs[self.MNq,self.MNj,:] + EiBjs[self.MNp,self.MNj,:]*Bis_Ejd[self.MNq,self.MNi,:])/(2*ell+1)
        # (2sDS)
        Ts_DS[2,:,:,:] = EiEjs[self.MNi,self.MNq,:]*Bis_Bjd[self.MNj,self.MNp,:]/(2*ell+1)
        # (2sDS*)
        Ts_DS[3,:,:,:] = EiEjs[self.MNp,self.MNj,:]*Bis_Bjd[self.MNq,self.MNi,:]/(2*ell+1)
        # (3sDS)
        Ts_DS[4,:,:,:] = BiBjs[self.MNi,self.MNq,:]*Eis_Ejd[self.MNj,self.MNp,:]/(2*ell+1)
        # (3sDS*)
        Ts_DS[5,:,:,:] = BiBjs[self.MNp,self.MNj,:]*Eis_Ejd[self.MNq,self.MNi,:]/(2*ell+1)
        # (4sDS) 
        Ts_DS[6,:,:,:] = Bis_Bjd[self.MNi,self.MNp,:]*EiEjs[self.MNj,self.MNq,:]/(2*ell+1)
        # (4sDS*) 
        Ts_DS[7,:,:,:] = Bis_Bjd[self.MNp,self.MNi,:]*EiEjs[self.MNq,self.MNj,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(Ts_DS, self.bin_conf), 3, 1)
        
    def C_dxsd(self, EiEjd, BiBjd, EiBjd, BiEjd, Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd):  
        lmax = self.spec.lmax
        ell = np.arange(0, lmax+1, 1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # dust * synch-dust 
        Td_SD=np.zeros((8, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1dSD)
        Td_SD[0,:,:,:] = (Eis_Ejd[self.MNp,self.MNi,:]*BiBjd[self.MNj,self.MNq,:] + EiBjd[self.MNi,self.MNq,:]*Eis_Bjd[self.MNp,self.MNj,:])/(2*ell+1)
        # (1dSD*)
        Td_SD[1,:,:,:] = (Eis_Ejd[self.MNi,self.MNp,:]*BiBjd[self.MNq,self.MNj,:] + EiBjd[self.MNp,self.MNj,:]*Eis_Bjd[self.MNi,self.MNq,:])/(2*ell+1)
        # (2dSD)
        Td_SD[2,:,:,:] = EiEjd[self.MNi,self.MNq,:]*Bis_Bjd[self.MNp,self.MNj,:]/(2*ell+1)
        # (2dSD*)
        Td_SD[3,:,:,:] = EiEjd[self.MNp,self.MNj,:]*Bis_Bjd[self.MNi,self.MNq,:]/(2*ell+1)
        # (3dSD)
        Td_SD[4,:,:,:] = BiBjd[self.MNi,self.MNq,:]*Eis_Ejd[self.MNp,self.MNj,:]/(2*ell+1)
        # (3dSD*)
        Td_SD[5,:,:,:] = BiBjd[self.MNp,self.MNj,:]*Eis_Ejd[self.MNi,self.MNq,:]/(2*ell+1)
        # (4dSD)
        Td_SD[6,:,:,:] = Bis_Bjd[self.MNp,self.MNi,:]*EiEjd[self.MNj,self.MNq,:]/(2*ell+1)
        # (4dSD*)
        Td_SD[7,:,:,:] = Bis_Bjd[self.MNi,self.MNp,:]*EiEjd[self.MNq,self.MNj,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(Td_SD, self.bin_conf), 3, 1)

    def C_dxds(self, EiEjd, BiBjd, EiBjd, BiEjd, Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd):  
        lmax = self.spec.lmax
        ell  = np.arange(0, lmax+1, 1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # dust * dust-synch 
        Td_DS=np.zeros((8, self.Nbands*(self.Nbands-self.avoid), self.Nbands*(self.Nbands-self.avoid), lmax+1), dtype=np.float64)
        # (1dDS)
        Td_DS[0,:,:,:] = (EiEjd[self.MNi,self.MNp,:]*Bis_Bjd[self.MNq,self.MNj,:] + Bis_Ejd[self.MNq,self.MNi,:]*BiEjd[self.MNj,self.MNp,:])/(2*ell+1)
        # (1dDS*)
        Td_DS[1,:,:,:] = (EiEjd[self.MNp,self.MNi,:]*Bis_Bjd[self.MNj,self.MNq,:] + Bis_Ejd[self.MNj,self.MNp,:]*BiEjd[self.MNq,self.MNi,:])/(2*ell+1)
        # (2dDS)
        Td_DS[2,:,:,:] = Eis_Ejd[self.MNq,self.MNi,:]*BiBjd[self.MNj,self.MNp,:]/(2*ell+1)
        # (2dDS*)
        Td_DS[3,:,:,:] = Eis_Ejd[self.MNj,self.MNp,:]*BiBjd[self.MNq,self.MNi,:]/(2*ell+1)
        # (3dDS)
        Td_DS[4,:,:,:] = Bis_Bjd[self.MNq,self.MNi,:]*EiEjd[self.MNj,self.MNp,:]/(2*ell+1)
        # (3dDS*)
        Td_DS[5,:,:,:] = Bis_Bjd[self.MNj,self.MNp,:]*EiEjd[self.MNq,self.MNi,:]/(2*ell+1)
        # (4dDS)
        Td_DS[6,:,:,:] = BiBjd[self.MNi,self.MNp,:]*Eis_Ejd[self.MNq,self.MNj,:]/(2*ell+1)
        # (4dDS*)
        Td_DS[7,:,:,:] = BiBjd[self.MNp,self.MNi,:]*Eis_Ejd[self.MNj,self.MNq,:]/(2*ell+1)
        return np.moveaxis(bin_cov_matrix(Td_DS, self.bin_conf), 3, 1)

############################################################################### 
### Format cls and calculate elements of covariance matrix

    def process_cls(self, incls): 
        lmax   = self.spec.lmax
        # common to all fits (basic alpha*alpha fit)
        EEo_ij_b = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
        BBo_ij_b = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64) 
        EBo_ij_b = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
        # observed * observed covariance
        EiEj_o   = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
        BiBj_o   = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
        EiBj_o   = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
        BiEj_o   = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
        # only used in particular combinations 
        if 'beta' in self.fit:
            EEcmb_ij_b = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            BBcmb_ij_b = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64) 
        if 'Ad' in self.fit:
            EBd_ij_b   = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            # dust - dust covariance
            EiEj_d     = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            BiBj_d     = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            EiBj_d     = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64) 
            BiEj_d     = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            # dust - obs covariance
            Eid_Ejo    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            Bid_Bjo    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            Eid_Bjo    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            Bid_Ejo    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
        if 'As' in self.fit:
            EBs_ij_b   = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            # sync - sync covariance
            EiEj_s     = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            BiBj_s     = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            EiBj_s     = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64) 
            BiEj_s     = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            # sync - obs covariance
            Eis_Ejo    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            Bis_Bjo    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            Eis_Bjo    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            Bis_Ejo    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            # sync - dust covariance
            Eis_Ejd    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            Bis_Bjd    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            Eis_Bjd    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            Bis_Ejd    = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
        if 'Asd' in self.fit:
            EsBd_ij_b  = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
            EdBs_ij_b  = np.zeros((self.Nbands, self.Nbands, lmax+1), dtype=np.float64)
 
        # format cls
        for ii, band_i in enumerate(self.bands):
            idx_i  = self.inst[band_i]['cl idx']
            freq_i = np.where(self.spec.lat.freqs==band_i[:-2])[0][0]
            fwhm_i = self.inst[band_i]['fwhm']
            for jj, band_j in enumerate(self.bands):
                idx_j  = self.inst[band_j]['cl idx']
                freq_j = np.where(self.spec.lat.freqs==band_j[:-2])[0][0]
                fwhm_j = self.inst[band_j]['fwhm']
                
                # common to all fits (basic alpha*alpha fit)
                EEo_ij_b[ii,jj,:] = incls['oxo'][idx_i,  idx_j,  0, :lmax+1]
                BBo_ij_b[ii,jj,:] = incls['oxo'][idx_i,  idx_j,  1, :lmax+1]
                EBo_ij_b[ii,jj,:] = incls['oxo'][idx_i,  idx_j,  2, :lmax+1]
                # observed * observed covariance
                EiEj_o[ii,jj,:]   = incls['oxo'][idx_i, idx_j, 0, :lmax+1]
                BiBj_o[ii,jj,:]   = incls['oxo'][idx_i, idx_j, 1, :lmax+1]
                EiBj_o[ii,jj,:]   = incls['oxo'][idx_i, idx_j, 2, :lmax+1]
                BiEj_o[ii,jj,:]   = incls['oxo'][idx_j, idx_i, 2, :lmax+1]
                # only used in particular combinations 
                if 'beta' in self.fit:
                    EEcmb_ij_b[ii,jj,:] = self.convolve_gaussBeams_pwf("ee", fwhm_i, fwhm_j, lmax)
                    BBcmb_ij_b[ii,jj,:] = self.convolve_gaussBeams_pwf("bb", fwhm_i, fwhm_j, lmax)
                if 'Ad' in self.fit:
                    EBd_ij_b[ii,jj,:]   = incls['dxd'][freq_i, freq_j, 2, :lmax+1]
                    # dust * dust covariance
                    EiEj_d[ii,jj,:]     = incls['dxd'][freq_i, freq_j, 0, :lmax+1]
                    BiBj_d[ii,jj,:]     = incls['dxd'][freq_i, freq_j, 1, :lmax+1]
                    EiBj_d[ii,jj,:]     = incls['dxd'][freq_i, freq_j, 2, :lmax+1]
                    BiEj_d[ii,jj,:]     = incls['dxd'][freq_j, freq_i, 2, :lmax+1]
                    # dust * obs covariance
                    Eid_Ejo[ii,jj,:]    = incls['dxo'][freq_i, idx_j, 0, :lmax+1]
                    Bid_Bjo[ii,jj,:]    = incls['dxo'][freq_i, idx_j, 1, :lmax+1]
                    Eid_Bjo[ii,jj,:]    = incls['dxo'][freq_i, idx_j, 2, :lmax+1]
                    Bid_Ejo[ii,jj,:]    = incls['dxo'][freq_i, idx_j, 3, :lmax+1]
                if 'As' in self.fit:
                    EBs_ij_b[ii,jj,:]   = incls['sxs'][freq_i, freq_j, 2, :lmax+1]
                    # sync * sync covariance
                    EiEj_s[ii,jj,:]     = incls['sxs'][freq_i, freq_j, 0, :lmax+1]
                    BiBj_s[ii,jj,:]     = incls['sxs'][freq_i, freq_j, 1, :lmax+1]
                    EiBj_s[ii,jj,:]     = incls['sxs'][freq_i, freq_j, 2, :lmax+1]
                    BiEj_s[ii,jj,:]     = incls['sxs'][freq_j, freq_i, 2, :lmax+1]
                    # sync * dust covariance
                    Eis_Ejd[ii,jj,:]    = incls['sxd'][freq_i, freq_j, 0, :lmax+1]
                    Bis_Bjd[ii,jj,:]    = incls['sxd'][freq_i, freq_j, 1, :lmax+1]
                    Eis_Bjd[ii,jj,:]    = incls['sxd'][freq_i, freq_j, 2, :lmax+1]
                    Bis_Ejd[ii,jj,:]    = incls['sxd'][freq_i, freq_j, 3, :lmax+1]
                    # sync * obs covariance
                    Eis_Ejo[ii,jj,:]    = incls['sxo'][freq_i, idx_j, 0, :lmax+1]
                    Bis_Bjo[ii,jj,:]    = incls['sxo'][freq_i, idx_j, 1, :lmax+1]
                    Eis_Bjo[ii,jj,:]    = incls['sxo'][freq_i, idx_j, 2, :lmax+1]
                    Bis_Ejo[ii,jj,:]    = incls['sxo'][freq_i, idx_j, 3, :lmax+1]
                if 'Asd' in self.fit:
                    EsBd_ij_b[ii,jj,:]  = incls['sxd'][freq_i, freq_j, 2, :lmax+1]
                    EdBs_ij_b[ii,jj,:]  = incls['sxd'][freq_j, freq_i, 3, :lmax+1]             

        # bin only once at the end
        self.bin_terms = {"EEo_ij_b":bin_spec_matrix(EEo_ij_b, self.bin_conf),
                          "BBo_ij_b":bin_spec_matrix(BBo_ij_b, self.bin_conf),
                          "EBo_ij_b":bin_spec_matrix(EBo_ij_b, self.bin_conf)}
        self.cov_terms = {"C_oxo":self.C_oxo(EiEj_o, BiBj_o, EiBj_o, BiEj_o)}
        # only used in particular combinations 
        if 'beta' in self.fit:
            self.bin_terms["EEcmb_ij_b"] = bin_spec_matrix(EEcmb_ij_b, self.bin_conf)
            self.bin_terms["BBcmb_ij_b"] = bin_spec_matrix(BBcmb_ij_b, self.bin_conf)
            self.cov_terms["C_cmb"]      = self.C_cmb()
        if 'Ad' in self.fit:
            self.bin_terms["EBd_ij_b"]   = bin_spec_matrix(EBd_ij_b, self.bin_conf)
            self.cov_terms["C_dxd"]      = self.C_fgxfg(EiEj_d, BiBj_d, EiBj_d, BiEj_d)
            self.cov_terms["C_dxo"]      = self.C_fgxo(Eid_Ejo, Bid_Bjo, Eid_Bjo, Bid_Ejo)
        if 'As' in self.fit:
            self.bin_terms["EBs_ij_b"]   = bin_spec_matrix(EBs_ij_b, self.bin_conf)
            self.cov_terms["C_sxs"]      = self.C_fgxfg(EiEj_s, BiBj_s, EiBj_s, BiEj_s)
            self.cov_terms["C_sxo"]      = self.C_fgxo(Eis_Ejo, Bis_Bjo, Eis_Bjo, Bis_Ejo)
            self.cov_terms["C_sxd"]      = self.C_sxd(Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd)
        if 'Asd' in self.fit:
            self.bin_terms["EsBd_ij_b"]  = bin_spec_matrix(EsBd_ij_b, self.bin_conf)
            self.bin_terms["EdBs_ij_b"]  = bin_spec_matrix(EdBs_ij_b, self.bin_conf)
            self.cov_terms["C_sdxsd"]    = self.C_sdxsd(EiEj_s, BiBj_s, EiBj_s, BiEj_s,
                                                        EiEj_d, BiBj_d, EiBj_d, BiEj_d, 
                                                        Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd)
            self.cov_terms["C_dsxds"]    = self.C_dsxds(EiEj_s, BiBj_s, EiBj_s, BiEj_s, 
                                                        EiEj_d, BiBj_d, EiBj_d, BiEj_d, 
                                                        Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd)
            self.cov_terms["C_sdxo"]     = self.C_sdxo(Eis_Ejo, Bis_Bjo, Eis_Bjo, Bis_Ejo, 
                                                       Eid_Ejo, Bid_Bjo, Eid_Bjo, Bid_Ejo)
            self.cov_terms["C_dsxo"]     = self.C_dsxo(Eis_Ejo, Bis_Bjo, Eis_Bjo, Bis_Ejo, 
                                                       Eid_Ejo, Bid_Bjo, Eid_Bjo, Bid_Ejo)
            self.cov_terms["C_sdxds"]    = self.C_sdxds(EiEj_s, BiBj_s, EiBj_s, BiEj_s, 
                                                        EiEj_d, BiBj_d, EiBj_d, BiEj_d, 
                                                        Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd)
            self.cov_terms["C_sxsd"]     = self.C_sxsd(EiEj_s, BiBj_s, EiBj_s, BiEj_s, Eis_Ejd, 
                                                       Bis_Bjd, Eis_Bjd, Bis_Ejd)
            self.cov_terms["C_sxds"]     = self.C_sxds(EiEj_s, BiBj_s, EiBj_s, BiEj_s, 
                                                       Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd)
            self.cov_terms["C_dxsd"]     = self.C_dxsd(EiEj_d, BiBj_d, EiBj_d, BiEj_d, 
                                                       Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd)
            self.cov_terms["C_dxds"]     = self.C_dxds(EiEj_d, BiBj_d, EiBj_d, BiEj_d, 
                                                       Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd)


###############################################################################    
### solve linear system to calculate maximum likelihood solution

    def solve_linear_system(self, iC, Niter, res):
        if self.fit=="alpha":
            return self.__linear_system_alpha__(iC, Niter, res)
        elif self.fit=="Ad + alpha":
            return self.__linear_system_Ad_alpha__(iC, Niter, res) 
        elif self.fit=="beta + alpha":
            return self.__linear_system_beta_alpha__(iC, Niter, res)
        elif self.fit=="As + Ad + alpha":
            return self.__linear_system_As_Ad_alpha__(iC, Niter, res)
        elif self.fit=="Ad + beta + alpha":
            return self.__linear_system_Ad_beta_alpha__(iC, Niter, res)
        elif self.fit=="As + Ad + beta + alpha":
            return self.__linear_system_As_Ad_beta_alpha__(iC, Niter, res)
        elif self.fit=="As + Asd + Ad + alpha":
            return self.__linear_system_As_Asd_Ad_alpha__(iC, Niter, res)
        elif self.fit=="As + Asd + Ad + beta + alpha":
            return self.__linear_system_As_Asd_Ad_beta_alpha__(iC, Niter, res)

    def __linear_system_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self, iC)
        linsys.compute_terms()
        # build system matrix and independent term
        sys_mat  = np.zeros((res.Nvar, res.Nvar), dtype=np.float64)
        ind_term = np.zeros(res.Nvar, dtype=np.float64)
        # variables ordered as alpha_i
        for ii, band_i in enumerate(self.bands):
            idx_i = self.inst[band_i]['alpha idx']
            # alpha_i
            ind_term[idx_i] += 2*(np.sum(linsys.D_ij[:,ii]) - np.sum(linsys.H_ij[ii,:]))
            for jj, band_j in enumerate(self.bands):
                idx_j = self.inst[band_j]['alpha idx']
                # alpha_i - alpha_j terms
                aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii]) + np.sum(linsys.E_ijpq[:, ii, :, jj])
                aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :]) + np.sum(linsys.B_ijpq[ii, :, jj, :])
                aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii]) + np.sum(linsys.I_ijpq[ii, :, :, jj])
                sys_mat[idx_i, idx_j] += 2*( aux1 + aux2 - 2*aux3 )
        # solve Ax=B
        # ang_now = np.matmul(np.linalg.pinv(sys_mat), ind_term) # risky alternative
        ang_now = np.linalg.solve(sys_mat, ind_term)
        cov_now = np.linalg.inv(sys_mat)
        std_now = np.sqrt(np.diagonal(cov_now)) 
        # save results even if something went wrong
        res.ml[f"Iter {Niter+1}"]         = {}
        res.std_fisher[f"Iter {Niter+1}"] = {}
        res.cov_fisher[f"Iter {Niter+1}"] = cov_now
        if self.alpha_per_split:
            for ii, band in enumerate(self.bands):
                res.ml[f"Iter {Niter+1}"][band]         = ang_now[ii]
                res.std_fisher[f"Iter {Niter+1}"][band] = std_now[ii]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                res.ml[f"Iter {Niter+1}"][freq]         = ang_now[ii]
                res.std_fisher[f"Iter {Niter+1}"][freq] = std_now[ii]
        if np.any( np.isnan(std_now) ):
            raise StopIteration()

    def __linear_system_Ad_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self, iC)
        linsys.compute_terms()
        # build system matrix and independent term
        sys_mat  = np.zeros((res.Nvar, res.Nvar), dtype=np.float64)
        ind_term = np.zeros(res.Nvar, dtype=np.float64)
        
        # variables ordered as Ad, alpha_i
        sys_mat[0, 0] = linsys.R[0] # Ad - Ad  
        ind_term[0]   = linsys.N[0] # Ad
        for ii, band_i in enumerate(self.bands):
            idx_i = self.inst[band_i]['alpha idx']
            
            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii]) - np.sum(linsys.omega_ij[ii,:])
            sys_mat[0, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[idx_i+res.ext_par, 0] += 2*Ad_ai

            ind_term[idx_i+res.ext_par] += 2*(np.sum(linsys.D_ij[:,ii]) - np.sum(linsys.H_ij[ii,:])) # alpha_i
            for jj, band_j in enumerate(self.bands):
                idx_j = self.inst[band_j]['alpha idx']
                # alpha_i - alpha_j terms
                aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii]) + np.sum(linsys.E_ijpq[:, ii, :, jj])
                aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :]) + np.sum(linsys.B_ijpq[ii, :, jj, :])
                aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii]) + np.sum(linsys.I_ijpq[ii, :, :, jj])
                sys_mat[idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
        
        # solve Ax=B
        # ang_now = np.matmul(np.linalg.pinv(sys_mat), ind_term) # risky alternative
        ang_now = np.linalg.solve(sys_mat, ind_term)
        cov_now = np.linalg.inv(sys_mat)
        std_now = np.sqrt(np.diagonal(cov_now)) 
        # save results even if something went wrong
        res.ml[f"Iter {Niter+1}"]         = {"Ad":ang_now[0]}
        res.std_fisher[f"Iter {Niter+1}"] = {"Ad":std_now[0]}
        res.cov_fisher[f"Iter {Niter+1}"] = cov_now
        if self.alpha_per_split:
            for ii, band in enumerate(self.bands):
                res.ml[f"Iter {Niter+1}"][band]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][band] = std_now[ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                res.ml[f"Iter {Niter+1}"][freq]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][freq] = std_now[ii+res.ext_par]
        if np.any( np.isnan(std_now) ):
            raise StopIteration()

    def __linear_system_As_Ad_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self, iC)
        linsys.compute_terms()
        # build system matrix and independent term
        sys_mat  = np.zeros((res.Nvar, res.Nvar),dtype=np.float64)
        ind_term = np.zeros(res.Nvar, dtype=np.float64)
        
        # variables ordered as As, Ad, Asd, beta, alpha_i
        sys_mat[0, 0] = linsys.S[0]                              # As - As
        sys_mat[0, 1] = linsys.W[0]; sys_mat[1, 0] = linsys.W[0] # As - Ad
        ind_term[0]   = linsys.J[0]                              # As

        sys_mat[1, 1] = linsys.R[0]                              # Ad - Ad
        ind_term[1]   = linsys.N[0]                              # Ad
        
        for ii, band_i in enumerate(self.bands):
            idx_i = self.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii]) - np.sum(linsys.psi_ij[ii,:])
            sys_mat[0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii]) - np.sum(linsys.omega_ij[ii,:])
            sys_mat[1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[idx_i+res.ext_par, 1] += 2*Ad_ai

            # alpha_i
            ind_term[idx_i+res.ext_par] += 2*(np.sum(linsys.D_ij[:,ii]) - np.sum(linsys.H_ij[ii,:]))
            
            for jj, band_j in enumerate(self.bands):
                idx_j = self.inst[band_j]['alpha idx']
                # alpha_i - alpha_j terms
                aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii]) + np.sum(linsys.E_ijpq[:, ii, :, jj])
                aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :]) + np.sum(linsys.B_ijpq[ii, :, jj, :])
                aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii]) + np.sum(linsys.I_ijpq[ii, :, :, jj])
                sys_mat[idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )

        # solve Ax=B
        # ang_now = np.matmul(np.linalg.pinv(sys_mat), ind_term) # risky alternative
        ang_now = np.linalg.solve(sys_mat, ind_term)
        cov_now = np.linalg.inv(sys_mat)
        std_now = np.sqrt(np.diagonal(cov_now)) 
        # save results even if something went wrong
        res.ml[f"Iter {Niter+1}"]         = {"As":ang_now[0], "Ad":ang_now[1]}
        res.std_fisher[f"Iter {Niter+1}"] = {"As":std_now[0], "Ad":std_now[1]}
        res.cov_fisher[f"Iter {Niter+1}"] = cov_now
        if self.alpha_per_split:
            for ii, band in enumerate(self.bands):
                res.ml[f"Iter {Niter+1}"][band]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][band] = std_now[ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                res.ml[f"Iter {Niter+1}"][freq]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][freq] = std_now[ii+res.ext_par]
        if np.any( np.isnan(std_now) ):
            raise StopIteration()

    def __linear_system_As_Asd_Ad_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self, iC)
        linsys.compute_terms()
        # build system matrix and independent term
        sys_mat  = np.zeros((res.Nvar, res.Nvar),dtype=np.float64)
        ind_term = np.zeros(res.Nvar, dtype=np.float64)
        
        # variables ordered as As, Ad, Asd, beta, alpha_i
        sys_mat[0, 0] = linsys.S[0]                               # As - As
        sys_mat[0, 1] = linsys.W[0]; sys_mat[1, 0] = linsys.W[0]  # As - Ad
        sys_mat[0, 2] = linsys.Q[0] + linsys.V[0]                 # As - Asd
        sys_mat[2, 0] = linsys.Q[0] + linsys.V[0]  
        ind_term[0]   = linsys.J[0]                               # As

        sys_mat[1, 1] = linsys.R[0]                               # Ad - Ad
        sys_mat[1, 2] = linsys.K[0] + linsys.xi[0]                # Ad - Asd
        sys_mat[2, 1] = linsys.K[0] + linsys.xi[0] 
        ind_term[1]   = linsys.N[0]                               # Ad
        
        sys_mat[2, 2] = linsys.T[0] + linsys.U[0] + 2*linsys.Z[0] # Asd - Asd
        ind_term[2]   = linsys.M[0] + linsys.L[0]                 # Asd
                                                  
        
        for ii, band_i in enumerate(self.bands):
            idx_i = self.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii]) - np.sum(linsys.psi_ij[ii,:])
            sys_mat[0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii]) - np.sum(linsys.omega_ij[ii,:])
            sys_mat[1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[idx_i+res.ext_par, 1] += 2*Ad_ai

            # Asd - alpha_i
            Asd_ai = np.sum(linsys.pi_ij[:,ii]) + np.sum(linsys.rho_ij[:,ii]) - np.sum(linsys.phi_ij[ii,:]) - np.sum(linsys.OMEGA_ij[ii,:])
            sys_mat[2, idx_i+res.ext_par] += 2*Asd_ai
            sys_mat[idx_i+res.ext_par, 2] += 2*Asd_ai

            # alpha_i
            ind_term[idx_i+res.ext_par] += 2*(np.sum(linsys.D_ij[:,ii]) - np.sum(linsys.H_ij[ii,:]))
            
            for jj, band_j in enumerate(self.bands):
                idx_j = self.inst[band_j]['alpha idx']
                # alpha_i - alpha_j terms
                aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii]) + np.sum(linsys.E_ijpq[:, ii, :, jj])
                aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :]) + np.sum(linsys.B_ijpq[ii, :, jj, :])
                aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii]) + np.sum(linsys.I_ijpq[ii, :, :, jj])
                sys_mat[idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )

        # solve Ax=B
        # ang_now = np.matmul(np.linalg.pinv(sys_mat), ind_term) # risky alternative
        ang_now = np.linalg.solve(sys_mat, ind_term)
        cov_now = np.linalg.inv(sys_mat)
        std_now = np.sqrt(np.diagonal(cov_now)) 
        # save results even if something went wrong
        res.ml[f"Iter {Niter+1}"]         = {"As":ang_now[0], "Ad":ang_now[1], "Asd":ang_now[2]}
        res.std_fisher[f"Iter {Niter+1}"] = {"As":std_now[0], "Ad":std_now[1], "Asd":std_now[2]}
        res.cov_fisher[f"Iter {Niter+1}"] = cov_now
        if self.alpha_per_split:
            for ii, band in enumerate(self.bands):
                res.ml[f"Iter {Niter+1}"][band]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][band] = std_now[ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                res.ml[f"Iter {Niter+1}"][freq]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][freq] = std_now[ii+res.ext_par]
        if np.any( np.isnan(std_now) ):
            raise StopIteration()

    def __linear_system_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self, iC)
        linsys.compute_terms()
        # build system matrix and independent term
        sys_mat  = np.zeros((res.Nvar, res.Nvar), dtype=np.float64)
        ind_term = np.zeros(res.Nvar, dtype=np.float64)
        
        # variables ordered as beta, alpha_i        
        sys_mat[0, 0] = 4*(linsys.G[0] + linsys.F[0] - 2*linsys.C[0]) # beta - beta
        ind_term[0]   = 2*(linsys.O[0] - linsys.P[0])                 # beta
        
        for ii, band_i in enumerate(self.bands):
            idx_i = self.inst[band_i]['alpha idx']
            
            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii]) + np.sum(linsys.epsilon_ij[ii,:]) - np.sum(linsys.varphi_ij[:,ii]) - np.sum(linsys.ene_ij[ii,:])
            sys_mat[0, idx_i+res.ext_par] += 4*b_a
            sys_mat[idx_i+res.ext_par, 0] += 4*b_a

            ind_term[idx_i+res.ext_par] += 2*(np.sum(linsys.D_ij[:,ii]) - np.sum(linsys.H_ij[ii,:])) # alpha_i
            for jj, band_j in enumerate(self.bands):
                idx_j = self.inst[band_j]['alpha idx']
                # alpha_i - alpha_j terms
                aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii]) + np.sum(linsys.E_ijpq[:, ii, :, jj])
                aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :]) + np.sum(linsys.B_ijpq[ii, :, jj, :])
                aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii]) + np.sum(linsys.I_ijpq[ii, :, :, jj])
                sys_mat[idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
        
        # solve Ax=B
        # ang_now = np.matmul(np.linalg.pinv(sys_mat), ind_term) # risky alternative
        ang_now = np.linalg.solve(sys_mat, ind_term)
        cov_now = np.linalg.inv(sys_mat)
        std_now = np.sqrt(np.diagonal(cov_now)) 
        # save results even if something went wrong
        res.ml[f"Iter {Niter+1}"]         = {"beta":ang_now[0]}
        res.std_fisher[f"Iter {Niter+1}"] = {"beta":std_now[0]}
        res.cov_fisher[f"Iter {Niter+1}"] = cov_now
        if self.alpha_per_split:
            for ii, band in enumerate(self.bands):
                res.ml[f"Iter {Niter+1}"][band]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][band] = std_now[ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                res.ml[f"Iter {Niter+1}"][freq]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][freq] = std_now[ii+res.ext_par]
        if np.any( np.isnan(std_now) ):
            raise StopIteration()
            
    def __linear_system_Ad_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self, iC)
        linsys.compute_terms()
        # build system matrix and independent term
        sys_mat  = np.zeros((res.Nvar, res.Nvar), dtype=np.float64)
        ind_term = np.zeros(res.Nvar, dtype=np.float64)
        
        # variables ordered as Ad, beta, alpha_i
        sys_mat[0, 0] = linsys.R[0]                         # Ad - Ad  
        sys_mat[0, 1] = 2*(linsys.LAMBDA[0] - linsys.mu[0]) # Ad - beta
        sys_mat[1, 0] = 2*(linsys.LAMBDA[0] - linsys.mu[0])  
        ind_term[0]   = linsys.N[0]                         # Ad
        
        sys_mat[1, 1] = 4*(linsys.G[0] + linsys.F[0] - 2*linsys.C[0]) # beta - beta
        ind_term[1]   = 2*(linsys.O[0] - linsys.P[0])                 # beta
        
        for ii, band_i in enumerate(self.bands):
            idx_i = self.inst[band_i]['alpha idx']
            
            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii]) - np.sum(linsys.omega_ij[ii,:])
            sys_mat[0, idx_i+res.ext_par] += 2*Ad_ai 
            sys_mat[idx_i+res.ext_par, 0] += 2*Ad_ai

            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii]) + np.sum(linsys.epsilon_ij[ii,:]) - np.sum(linsys.varphi_ij[:,ii]) - np.sum(linsys.ene_ij[ii,:])
            sys_mat[1, idx_i+res.ext_par] += 4*b_a
            sys_mat[idx_i+res.ext_par, 1] += 4*b_a

            ind_term[idx_i+res.ext_par] += 2*(np.sum(linsys.D_ij[:,ii]) - np.sum(linsys.H_ij[ii,:])) # alpha_i
            for jj, band_j in enumerate(self.bands):
                idx_j = self.inst[band_j]['alpha idx']
                # alpha_i - alpha_j terms
                aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii]) + np.sum(linsys.E_ijpq[:, ii, :, jj])
                aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :]) + np.sum(linsys.B_ijpq[ii, :, jj, :])
                aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii]) + np.sum(linsys.I_ijpq[ii, :, :, jj])
                sys_mat[idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
        
        # solve Ax=B
        # ang_now = np.matmul(np.linalg.pinv(sys_mat), ind_term) # risky alternative
        ang_now = np.linalg.solve(sys_mat, ind_term)
        cov_now = np.linalg.inv(sys_mat)
        std_now = np.sqrt(np.diagonal(cov_now)) 
        # save results even if something went wrong
        res.ml[f"Iter {Niter+1}"]         = {"Ad":ang_now[0], "beta":ang_now[1]}
        res.std_fisher[f"Iter {Niter+1}"] = {"Ad":std_now[0], "beta":std_now[1]}
        res.cov_fisher[f"Iter {Niter+1}"] = cov_now
        if self.alpha_per_split:
            for ii, band in enumerate(self.bands):
                res.ml[f"Iter {Niter+1}"][band]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][band] = std_now[ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                res.ml[f"Iter {Niter+1}"][freq]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][freq] = std_now[ii+res.ext_par]
        if np.any( np.isnan(std_now) ):
            raise StopIteration()
            
    def __linear_system_As_Ad_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self, iC)
        linsys.compute_terms()
        # build system matrix and independent term
        sys_mat  = np.zeros((res.Nvar, res.Nvar),dtype=np.float64)
        ind_term = np.zeros(res.Nvar, dtype=np.float64)
        
        # variables ordered as As, Ad, Asd, beta, alpha_i
        sys_mat[0, 0] = linsys.S[0]                                # As - As
        sys_mat[0, 1] = linsys.W[0]; sys_mat[1, 0] = linsys.W[0]   # As - Ad
        sys_mat[0, 2] = 2*(linsys.X[0] - linsys.Y[0])              # As - beta
        sys_mat[2, 0] = 2*(linsys.X[0] - linsys.Y[0]) 
        ind_term[0]   = linsys.J[0]                                # As

        sys_mat[1, 1] = linsys.R[0]                                # Ad - Ad
        sys_mat[1, 2] = 2*(linsys.LAMBDA[0] - linsys.mu[0])        # Ad - beta
        sys_mat[2, 1] = 2*(linsys.LAMBDA[0] - linsys.mu[0]) 
        ind_term[1]   = linsys.N[0]                                # Ad
        
        sys_mat[2, 2] = 4*(linsys.G[0] + linsys.F[0] - 2*linsys.C[0]) # beta - beta
        ind_term[2]   = 2*(linsys.O[0] - linsys.P[0])                 # beta
        
        for ii, band_i in enumerate(self.bands):
            idx_i = self.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii]) - np.sum(linsys.psi_ij[ii,:])
            sys_mat[0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii]) - np.sum(linsys.omega_ij[ii,:])
            sys_mat[1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[idx_i+res.ext_par, 1] += 2*Ad_ai

            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii]) + np.sum(linsys.epsilon_ij[ii,:]) - np.sum(linsys.varphi_ij[:,ii]) - np.sum(linsys.ene_ij[ii,:])
            sys_mat[2, idx_i+res.ext_par] += 4*b_a
            sys_mat[idx_i+res.ext_par, 2] += 4*b_a

            # alpha_i
            ind_term[idx_i+res.ext_par] += 2*(np.sum(linsys.D_ij[:,ii]) - np.sum(linsys.H_ij[ii,:]))
            
            for jj, band_j in enumerate(self.bands):
                idx_j = self.inst[band_j]['alpha idx']
                # alpha_i - alpha_j terms
                aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii]) + np.sum(linsys.E_ijpq[:, ii, :, jj])
                aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :]) + np.sum(linsys.B_ijpq[ii, :, jj, :])
                aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii]) + np.sum(linsys.I_ijpq[ii, :, :, jj])
                sys_mat[idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )

        # solve Ax=B
        # ang_now = np.matmul(np.linalg.pinv(sys_mat), ind_term) # risky alternative
        ang_now = np.linalg.solve(sys_mat, ind_term)
        cov_now = np.linalg.inv(sys_mat)
        std_now = np.sqrt(np.diagonal(cov_now)) 
        
        # save results even if something went wrong
        res.ml[f"Iter {Niter+1}"]         = {"As":ang_now[0], "Ad":ang_now[1], "beta":ang_now[2]}
        res.std_fisher[f"Iter {Niter+1}"] = {"As":std_now[0], "Ad":std_now[1], "beta":std_now[2]}
        res.cov_fisher[f"Iter {Niter+1}"] = cov_now
        if self.alpha_per_split:
            for ii, band in enumerate(self.bands):
                res.ml[f"Iter {Niter+1}"][band]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][band] = std_now[ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                res.ml[f"Iter {Niter+1}"][freq]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][freq] = std_now[ii+res.ext_par]
        if np.any( np.isnan(std_now) ):
            raise StopIteration()

    def __linear_system_As_Asd_Ad_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self, iC)
        linsys.compute_terms()
        # build system matrix and independent term
        sys_mat  = np.zeros((res.Nvar, res.Nvar),dtype=np.float64)
        ind_term = np.zeros(res.Nvar, dtype=np.float64)
        
        # variables ordered as As, Ad, Asd, beta, alpha_i
        sys_mat[0, 0] = linsys.S[0]                                # As - As
        sys_mat[0, 1] = linsys.W[0]; sys_mat[1, 0] = linsys.W[0]   # As - Ad
        sys_mat[0, 2] = linsys.Q[0] + linsys.V[0]                  # As - Asd
        sys_mat[2, 0] = linsys.Q[0] + linsys.V[0]     
        sys_mat[0, 3] = 2*(linsys.X[0] - linsys.Y[0])              # As - beta
        sys_mat[3, 0] = 2*(linsys.X[0] - linsys.Y[0]) 
        ind_term[0]   = linsys.J[0]                                # As

        sys_mat[1, 1] = linsys.R[0]                                # Ad - Ad
        sys_mat[1, 2] = linsys.K[0] + linsys.xi[0]                 # Ad - Asd   
        sys_mat[2, 1] = linsys.K[0] + linsys.xi[0]           
        sys_mat[1, 3] = 2*(linsys.LAMBDA[0] - linsys.mu[0])        # Ad - beta
        sys_mat[3, 1] = 2*(linsys.LAMBDA[0] - linsys.mu[0]) 
        ind_term[1]   = linsys.N[0]                                # Ad
        
        sys_mat[2, 2] = linsys.T[0] + linsys.U[0] + 2*linsys.Z[0]  # Asd - Asd
        sys_mat[2, 3] = 2*(linsys.DELTA[0] - linsys.delta[0] + linsys.eta[0] - linsys.theta[0]) # Asd - beta
        sys_mat[3, 2] = 2*(linsys.DELTA[0] - linsys.delta[0] + linsys.eta[0] - linsys.theta[0]) 
        ind_term[2]   = linsys.M[0] + linsys.L[0]                  # Asd
                                                           
        sys_mat[3, 3] = 4*(linsys.G[0] + linsys.F[0] - 2*linsys.C[0]) # beta - beta
        ind_term[3]   = 2*(linsys.O[0] - linsys.P[0])                 # beta
        
        for ii, band_i in enumerate(self.bands):
            idx_i = self.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii]) - np.sum(linsys.psi_ij[ii,:])
            sys_mat[0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii]) - np.sum(linsys.omega_ij[ii,:])
            sys_mat[1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[idx_i+res.ext_par, 1] += 2*Ad_ai

            # Asd - alpha_i
            Asd_ai = np.sum(linsys.pi_ij[:,ii]) + np.sum(linsys.rho_ij[:,ii]) - np.sum(linsys.phi_ij[ii,:]) - np.sum(linsys.OMEGA_ij[ii,:])
            sys_mat[2, idx_i+res.ext_par] += 2*Asd_ai
            sys_mat[idx_i+res.ext_par, 2] += 2*Asd_ai

            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii]) + np.sum(linsys.epsilon_ij[ii,:]) - np.sum(linsys.varphi_ij[:,ii]) - np.sum(linsys.ene_ij[ii,:])
            sys_mat[3, idx_i+res.ext_par] += 4*b_a
            sys_mat[idx_i+res.ext_par, 3] += 4*b_a

            # alpha_i
            ind_term[idx_i+res.ext_par] += 2*(np.sum(linsys.D_ij[:,ii]) - np.sum(linsys.H_ij[ii,:]))
            
            for jj, band_j in enumerate(self.bands):
                idx_j = self.inst[band_j]['alpha idx']
                # alpha_i - alpha_j terms
                aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii]) + np.sum(linsys.E_ijpq[:, ii, :, jj])
                aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :]) + np.sum(linsys.B_ijpq[ii, :, jj, :])
                aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii]) + np.sum(linsys.I_ijpq[ii, :, :, jj])
                sys_mat[idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )

        # solve Ax=B
        # ang_now = np.matmul(np.linalg.pinv(sys_mat), ind_term) # risky alternative
        ang_now = np.linalg.solve(sys_mat, ind_term)
        cov_now = np.linalg.inv(sys_mat)
        std_now = np.sqrt(np.diagonal(cov_now)) 
        # save results even if something went wrong
        res.ml[f"Iter {Niter+1}"]         = {"As":ang_now[0], "Ad":ang_now[1], "Asd":ang_now[2],"beta":ang_now[3]}
        res.std_fisher[f"Iter {Niter+1}"] = {"As":std_now[0], "Ad":std_now[1], "Asd":std_now[2],"beta":std_now[3]}
        res.cov_fisher[f"Iter {Niter+1}"] = cov_now
        if self.alpha_per_split:
            for ii, band in enumerate(self.bands):
                res.ml[f"Iter {Niter+1}"][band]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][band] = std_now[ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                res.ml[f"Iter {Niter+1}"][freq]         = ang_now[ii+res.ext_par]
                res.std_fisher[f"Iter {Niter+1}"][freq] = std_now[ii+res.ext_par]
        if np.any( np.isnan(std_now) ):
            raise StopIteration()

############################################################################## 
    
    def calculate(self, idx, return_result=False):
        # this function always saves the result
        res = Result(self.spec, self.fit, idx, self.alpha_per_split, 
                     self.rm_same_tube, self.nlb, self.bmin, self.bmax)
        # read the input spectra 
        try:
            input_cls = self.spec.get_spectra(idx, sync='As' in self.fit)
        except TypeError:
            self.spec.compute(self.sim_idx, sync='As' in self.fit)
            input_cls = self.spec.get_spectra(idx, sync='As' in self.fit)
        
        # format cls and calculate elements of covariance matrix
        self.process_cls(input_cls)
        del input_cls # free memory

        converged = False
        niter     = 0
        while not converged:
            cov    = self.build_cov(niter, res)
            invcov = np.linalg.inv(cov/self.fsky)
            try:
                self.solve_linear_system(invcov, niter, res)
                # evaluate convergence of the iterative calculation 
                # use only the angles as convergence criterion, not amplitude
                # use alpha + beta sum as convergence criterion 
                ang_now      = self.__get_ml_alphas__(niter+1, res, add_beta='beta' in self.fit)
                #difference with i-1
                ang_before_1 = self.__get_ml_alphas__(niter, res, add_beta='beta' in self.fit)
                c1           = np.abs(ang_now-ang_before_1)*rad2arcmin >= self.tol
                if np.sum(c1)<=1 or niter>self.niter_max:
                    converged = True
                elif niter>0:
                    #difference with i-2 
                    ang_before_2 = self.__get_ml_alphas__(niter-1, res, add_beta='beta' in self.fit)
                    c2 = np.abs(ang_now-ang_before_2)*rad2arcmin >= self.tol
                    if np.sum(c2)<=1:
                        converged = True
            except StopIteration:
                print('NaN in covariance')
                converged = True
                
            niter += 1
        #save results to disk 
        pl.dump(res, open(self.result_name(idx), "wb"), protocol=pl.HIGHEST_PROTOCOL)
        if return_result:
            return res
   
    def result_name(self, idx):
        path     = self.libdir
        fit_tag  = f"{self.fit.replace(' + ','_')}{'_sameAlphaPerSplit' if self.alpha_per_split else '_diffAlphaPerSplit'}{'_rmSameTube' if self.rm_same_tube else ''}{'_tempBP'if self.spec.temp_bp else ''}" 
        bin_tag  = f"Nb{self.nlb}_bmin{self.bmin}_bmax{self.bmax}"
        spec_tag = f"aposcale{str(self.spec.aposcale).replace('.','p')}{'_CO' if self.spec.CO else ''}{'_PS' if self.spec.PS else ''}{'_pureB' if self.spec.pureB else ''}_N{self.nside}"
        return f"{path}/ml_params_{fit_tag}_{bin_tag}_{spec_tag}_{idx:03d}.pkl"
       
    def estimate_angles(self, idx, overwrite=False, Niter=-1):
        file = self.result_name(idx)
        if (not os.path.isfile(file)) or overwrite:
            res = self.calculate(idx, return_result=True)
        else:
            res = pl.load(open(file, "rb"))
        max_iter = len(res.ml.keys())
        params   = {}
        for var in res.variables.split(", "):
            if var not in ["As", "Asd", "Ad"]:
                params[var] = np.rad2deg(res.ml[f"Iter {max_iter-1 if Niter==-1 else Niter}"][var])
            else:
                params[var] = res.ml[f"Iter {max_iter-1 if Niter==-1 else Niter}"][var]
        return  params
    
    
class S2N:
    
    def __init__(self, libdir, mode, nside, atm_noise, nsplits, dust, sync, 
                 template_bandpass, fit, bmin, bmax, alpha_per_split, rm_same_tube,
                 bandpass=True, aposcale=2.0, CO=True, PS=True, pureB=True,
                 window=5, binwidth=20):
        
        self.libdir    = libdir+'/S2N'
        os.makedirs(self.libdir, exist_ok=True)
        self.mode      = mode
        self.window    = window
        self.nside     = nside
        self.atm_noise = atm_noise
        self.dust      = dust
        self.sync      = sync
        self.nsplits   = nsplits
        self.bp        = bandpass
        self.temp_bp   = template_bandpass
        self.apo       = aposcale
        self.CO        = CO
        self.PS        = PS
        # create specific simulations for s/n calculation
        # fixed values for the calculation of the s/n
        self.beta            = 0.3 # deg
        self.alpha           = 0.3 # deg
        self.idx             = 0
        self.sky             = LATsky(self.libdir, nside, self.beta, dust, sync, 
                                self.alpha, atm_noise, nsplits, self.bp)
        self.sky.saveObsQUs(self.idx)
        self.spec            = Spectra(self.sky, template_bandpass=template_bandpass,
                                       pureB=pureB, aposcale=aposcale, CO=CO, PS=PS)
        self.fit             = fit
        self.alpha_per_split = alpha_per_split
        self.rm_same_tube    = rm_same_tube
        self.bmin            = bmin
        self.bmax            = bmax
        self.mle             = MLE(self.libdir, self.spec, fit, 
                                   alpha_per_split=alpha_per_split, 
                                   rm_same_tube=rm_same_tube, 
                                   bmax=bmax, bmin=bmin, binwidth=binwidth)
     
          
    def s2n_ell(self, iC, Niter, res):
        if self.fit=="alpha":
            return self.__s2n_ell_alpha__(iC, Niter, res)
        elif self.fit=="Ad + alpha":
            return self.__s2n_ell_Ad_alpha__(iC, Niter, res) 
        elif self.fit=="beta + alpha":
            return self.__s2n_ell_beta_alpha__(iC, Niter, res)
        elif self.fit=="As + Ad + alpha":
            return self.__s2n_ell_As_Ad_alpha__(iC, Niter, res)
        elif self.fit=="Ad + beta + alpha":
            return self.__s2n_ell_Ad_beta_alpha__(iC, Niter, res)
        elif self.fit=="As + Ad + beta + alpha":
            return self.__s2n_ell_As_Ad_beta_alpha__(iC, Niter, res)
        elif self.fit=="As + Asd + Ad + alpha":
            return self.__s2n_ell_As_Asd_Ad_alpha__(iC, Niter, res)
        elif self.fit=="As + Asd + Ad + beta + alpha":
            return self.__s2n_ell_As_Asd_Ad_beta_alpha__(iC, Niter, res)

    def __s2n_ell_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as alpha_i
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = {}
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
   
    def __s2n_ell_Ad_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as Ad, alpha_i
        sys_mat[:, 0, 0] = linsys.R  # Ad - Ad        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*Ad_ai            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,0]}
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
            
    def __s2n_ell_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as beta, alpha_i
        sys_mat[:, 0, 0] = 4*(linsys.G + linsys.F - 2*linsys.C) # beta - beta
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii,:], axis=0) + np.sum(linsys.epsilon_ij[ii,:,:], axis=0) - np.sum(linsys.varphi_ij[:,ii,:], axis=0) - np.sum(linsys.ene_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 4*b_a
            sys_mat[:, idx_i+res.ext_par, 0] += 4*b_a
            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "beta":res.ml[f"Iter {Niter}"]["beta"]/std_ell[:,3] }
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
            
    def __s2n_ell_As_Ad_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as As, Ad, alpha_i
        sys_mat[:, 0, 0] = linsys.S                              # As - As
        sys_mat[:, 0, 1] = linsys.W; sys_mat[:, 1, 0] = linsys.W # As - Ad

        sys_mat[:, 1, 1] = linsys.R                              # Ad - Ad
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii,:], axis=0) - np.sum(linsys.psi_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 1] += 2*Ad_ai

            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "As":  res.ml[f"Iter {Niter}"]["As"]  /std_ell[:,0],
                "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,1]}
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
                    
    def __s2n_ell_Ad_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as Ad, beta, alpha_i
        sys_mat[:, 0, 0] = linsys.R                              # Ad - Ad
        sys_mat[:, 0, 1] = 2*(linsys.LAMBDA - linsys.mu)         # Ad - beta
        sys_mat[:, 1, 0] = 2*(linsys.LAMBDA - linsys.mu) 
        
        sys_mat[:, 1, 1] = 4*(linsys.G + linsys.F - 2*linsys.C) # beta - beta
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*Ad_ai

            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii,:], axis=0) + np.sum(linsys.epsilon_ij[ii,:,:], axis=0) - np.sum(linsys.varphi_ij[:,ii,:], axis=0) - np.sum(linsys.ene_ij[ii,:,:], axis=0)
            sys_mat[:, 1, idx_i+res.ext_par] += 4*b_a
            sys_mat[:, idx_i+res.ext_par, 1] += 4*b_a
            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,1],
                "beta":res.ml[f"Iter {Niter}"]["beta"]/std_ell[:,3] }
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n

    def __s2n_ell_As_Ad_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as As, Ad, beta, alpha_i
        sys_mat[:, 0, 0] = linsys.S                              # As - As
        sys_mat[:, 0, 1] = linsys.W; sys_mat[:, 1, 0] = linsys.W # As - Ad
        sys_mat[:, 0, 2] = 2*(linsys.X - linsys.Y)               # As - beta
        sys_mat[:, 2, 0] = 2*(linsys.X - linsys.Y)

        sys_mat[:, 1, 1] = linsys.R                              # Ad - Ad
        sys_mat[:, 1, 2] = 2*(linsys.LAMBDA - linsys.mu)         # Ad - beta
        sys_mat[:, 2, 1] = 2*(linsys.LAMBDA - linsys.mu) 
        
        sys_mat[:, 2, 2] = 4*(linsys.G + linsys.F - 2*linsys.C) # beta - beta
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii,:], axis=0) - np.sum(linsys.psi_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 1] += 2*Ad_ai

            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii,:], axis=0) + np.sum(linsys.epsilon_ij[ii,:,:], axis=0) - np.sum(linsys.varphi_ij[:,ii,:], axis=0) - np.sum(linsys.ene_ij[ii,:,:], axis=0)
            sys_mat[:, 2, idx_i+res.ext_par] += 4*b_a
            sys_mat[:, idx_i+res.ext_par, 2] += 4*b_a
            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "As":  res.ml[f"Iter {Niter}"]["As"]  /std_ell[:,0],
                "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,1],
                "beta":res.ml[f"Iter {Niter}"]["beta"]/std_ell[:,2] }
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
      
    def __s2n_ell_As_Asd_Ad_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as As, Ad, Asd, alpha_i
        sys_mat[:, 0, 0] = linsys.S                              # As - As
        sys_mat[:, 0, 1] = linsys.W; sys_mat[:, 1, 0] = linsys.W # As - Ad
        sys_mat[:, 0, 2] = linsys.Q + linsys.V                   # As - Asd
        sys_mat[:, 2, 0] = linsys.Q + linsys.V     

        sys_mat[:, 1, 1] = linsys.R                              # Ad - Ad
        sys_mat[:, 1, 2] = linsys.K + linsys.xi                  # Ad - Asd   
        sys_mat[:, 2, 1] = linsys.K + linsys.xi           
        
        sys_mat[:, 2, 2] = linsys.T + linsys.U + 2*linsys.Z      # Asd - Asd
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii,:], axis=0) - np.sum(linsys.psi_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 1] += 2*Ad_ai

            # Asd - alpha_i
            Asd_ai = np.sum(linsys.pi_ij[:,ii,:], axis=0) + np.sum(linsys.rho_ij[:,ii,:], axis=0) - np.sum(linsys.phi_ij[ii,:,:], axis=0) - np.sum(linsys.OMEGA_ij[ii,:,:], axis=0)
            sys_mat[:, 2, idx_i+res.ext_par] += 2*Asd_ai
            sys_mat[:, idx_i+res.ext_par, 2] += 2*Asd_ai
            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "As":  res.ml[f"Iter {Niter}"]["As"]  /std_ell[:,0],
                "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,1],
                "Asd": res.ml[f"Iter {Niter}"]["Asd"] /std_ell[:,2]}
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n
    
    def __s2n_ell_As_Asd_Ad_beta_alpha__(self, iC, Niter, res):
        linsys = LinearSystem(self.mle, iC, mode=self.mode, window=self.window)
        linsys.compute_terms()
        # build system matrix
        sys_mat =np.zeros((linsys.ext_dim, res.Nvar, res.Nvar), dtype=np.float64)
        # variables ordered as As, Ad, Asd, beta, alpha_i
        sys_mat[:, 0, 0] = linsys.S                              # As - As
        sys_mat[:, 0, 1] = linsys.W; sys_mat[:, 1, 0] = linsys.W # As - Ad
        sys_mat[:, 0, 2] = linsys.Q + linsys.V                   # As - Asd
        sys_mat[:, 2, 0] = linsys.Q + linsys.V     
        sys_mat[:, 0, 3] = 2*(linsys.X - linsys.Y)               # As - beta
        sys_mat[:, 3, 0] = 2*(linsys.X - linsys.Y)

        sys_mat[:, 1, 1] = linsys.R                              # Ad - Ad
        sys_mat[:, 1, 2] = linsys.K + linsys.xi                  # Ad - Asd   
        sys_mat[:, 2, 1] = linsys.K + linsys.xi           
        sys_mat[:, 1, 3] = 2*(linsys.LAMBDA - linsys.mu)         # Ad - beta
        sys_mat[:, 3, 1] = 2*(linsys.LAMBDA - linsys.mu) 
        
        sys_mat[:, 2, 2] = linsys.T + linsys.U + 2*linsys.Z      # Asd - Asd
        sys_mat[:, 2, 3] = 2*(linsys.DELTA - linsys.delta + linsys.eta - linsys.theta) # Asd - beta
        sys_mat[:, 3, 2] = 2*(linsys.DELTA - linsys.delta + linsys.eta - linsys.theta) 
                                                           
        sys_mat[:, 3, 3] = 4*(linsys.G + linsys.F - 2*linsys.C) # beta - beta
        
        for ii, band_i in enumerate(self.mle.bands):
            idx_i = self.mle.inst[band_i]['alpha idx']
            
            # As - alpha_i 
            As_ai = np.sum(linsys.nu_ij[:,ii,:], axis=0) - np.sum(linsys.psi_ij[ii,:,:], axis=0)
            sys_mat[:, 0, idx_i+res.ext_par] += 2*As_ai
            sys_mat[:, idx_i+res.ext_par, 0] += 2*As_ai

            # Ad - alpha_i
            Ad_ai = np.sum(linsys.sigma_ij[:,ii,:], axis=0) - np.sum(linsys.omega_ij[ii,:,:], axis=0)
            sys_mat[:, 1, idx_i+res.ext_par] += 2*Ad_ai
            sys_mat[:, idx_i+res.ext_par, 1] += 2*Ad_ai

            # Asd - alpha_i
            Asd_ai = np.sum(linsys.pi_ij[:,ii,:], axis=0) + np.sum(linsys.rho_ij[:,ii,:], axis=0) - np.sum(linsys.phi_ij[ii,:,:], axis=0) - np.sum(linsys.OMEGA_ij[ii,:,:], axis=0)
            sys_mat[:, 2, idx_i+res.ext_par] += 2*Asd_ai
            sys_mat[:, idx_i+res.ext_par, 2] += 2*Asd_ai

            # beta - alpha_i
            b_a = np.sum(linsys.tau_ij[:,ii,:], axis=0) + np.sum(linsys.epsilon_ij[ii,:,:], axis=0) - np.sum(linsys.varphi_ij[:,ii,:], axis=0) - np.sum(linsys.ene_ij[ii,:,:], axis=0)
            sys_mat[:, 3, idx_i+res.ext_par] += 4*b_a
            sys_mat[:, idx_i+res.ext_par, 3] += 4*b_a
            
            for jj, band_j in enumerate(self.mle.bands):
                idx_j = self.mle.inst[band_j]['alpha idx']
                for ll in range(0, linsys.ext_dim, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(linsys.E_ijpq[:, jj, :, ii, ll]) + np.sum(linsys.E_ijpq[:, ii, :, jj, ll])
                    aux2 = np.sum(linsys.B_ijpq[jj, :, ii, :, ll]) + np.sum(linsys.B_ijpq[ii, :, jj, :, ll])
                    aux3 = np.sum(linsys.I_ijpq[jj, :, :, ii, ll]) + np.sum(linsys.I_ijpq[ii, :, :, jj, ll])
                    sys_mat[ll, idx_i+res.ext_par, idx_j+res.ext_par] += 2*( aux1 + aux2 - 2*aux3 )
                    
        # cov_ell = np.linalg.pinv(sys_mat) # risky alternative 
        cov_ell = np.linalg.inv(sys_mat)
        std_ell = np.sqrt(np.diagonal(cov_ell, axis1=1, axis2=2))
        if np.any( np.isnan(std_ell) ):
            raise StopIteration()
            
        s2n = { "As":  res.ml[f"Iter {Niter}"]["As"]  /std_ell[:,0],
                "Ad":  res.ml[f"Iter {Niter}"]["Ad"]  /std_ell[:,1],
                "Asd": res.ml[f"Iter {Niter}"]["Asd"] /std_ell[:,2],
                "beta":res.ml[f"Iter {Niter}"]["beta"]/std_ell[:,3] }
        if self.alpha_per_split:
            for ii, band in enumerate(self.mle.bands):
                s2n[band] = res.ml[f"Iter {Niter}"][band]/std_ell[:,ii+res.ext_par]
        else:
            for ii, freq in enumerate(self.spec.freqs):
                s2n[freq] = res.ml[f"Iter {Niter}"][freq]/std_ell[:,ii+res.ext_par]
                
        return s2n

    def calculate(self, fit=None, bmin=None, bmax=None, binwidth=None,
                  alpha_per_split=None, rm_same_tube=None):
        
        # reuse the object to calculate a new fit 
        if fit!=None or bmin!=None or bmax!=None or binwidth!=None or alpha_per_split!=None or rm_same_tube!=None: 
            if fit!=None:  self.fit = fit
            if bmin!=None: self.bmin = bmin
            if bmax!=None: self.bmax = bmax
            nlb = binwidth if binwidth!=None else self.mle.nlb
            if alpha_per_split!=None: self.alpha_per_split = alpha_per_split
            if rm_same_tube!=None:    self.rm_same_tube    = rm_same_tube
            self.mle = MLE(self.libdir, self.spec, self.fit, 
                           alpha_per_split=self.alpha_per_split, 
                           rm_same_tube=self.rm_same_tube, 
                           bmax=self.bmax, bmin=self.bmin, binwidth=nlb)
            
        niter = 0
        true_params = Result(self.spec, self.fit, self.idx, self.alpha_per_split, 
                             self.rm_same_tube, self.mle.nlb, self.bmin, self.bmax,
                             beta_ini=np.deg2rad(self.beta),
                             alpha_ini=np.deg2rad(self.alpha))
        # read the input spectra 
        try:
            input_cls = self.spec.get_spectra(self.idx, sync='As' in self.fit)
        except TypeError:
            self.spec.compute(self.idx, sync='As' in self.fit)
            input_cls = self.spec.get_spectra(self.idx, sync='As' in self.fit)
        
        # format cls and calculate elements of covariance matrix 
        self.mle.process_cls(input_cls)
        del input_cls # free memory
        cov    = self.mle.build_cov(niter, true_params)
        invcov = np.linalg.inv(cov/self.spec.fsky)
        # calculate signal-to-noise per multipole
        try:
            s2n = self.s2n_ell(invcov, niter, true_params)
        except StopIteration:
            print('NaN in covariance')
        # effective ell corresponding to each bin
        eff_ell = effective_ell(self.spec.lmax, self.mle.bin_conf)
        if self.mode=="ell":
            eff_ell = eff_ell[self.window//2:-self.window//2+1]
        return eff_ell, s2n
        
   
