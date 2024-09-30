
import numpy as np
from solat_cb.mle.utils import moving_sum


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
