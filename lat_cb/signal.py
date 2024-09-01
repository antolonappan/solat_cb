import numpy as np
import healpy as hp
import pysm3
from pysm3 import units as u
import camb
import os
import pickle as pl
from tqdm import tqdm
from lat_cb import mpi
import matplotlib.pyplot as plt

ini_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cb.ini")
spectra = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spectra.pkl")
mask = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mask_N1024.fits")
bp_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bp_profile.pkl")

def inrad(alpha):
    return np.deg2rad(alpha) # type: ignore

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

def synalm_c1(cl_x, alm_x, cl_y, cl_xy):
    lmax = hp.Alm.getlmax(len(alm_x))
    assert len(cl_x) >= lmax + 1, 'cl_x is less than the lmax of alm_x'
    assert len(cl_y) >= lmax + 1, 'cl_y is less than the lmax of alm_x'
    assert len(cl_xy) >= lmax + 1, 'cl_xy is less than the lmax of alm_x'
    corr = cl_xy * cli(cl_x)
    elm = hp.synalm(cl_y - corr*cl_xy, lmax=lmax)
    return elm + hp.almxfl(alm_x ,corr)

def SO_LAT_Nell(sensitivity_mode,f_sky,ell_max,sqrt=True):
    ## returns noise curves in both temperature and polarization, including the impact of the beam, for the SO large aperture telescope
    # sensitivity_mode:
    #     1: baseline, 
    #     2: goal
    # f_sky:  number from 0-1
    # ell_max: the maximum value of ell used in the computation of N(ell)
    ####################################################################
    ###                        Internal variables
    ## LARGE APERTURE
    # configuration
    # ensure valid parameter choices
    assert( sensitivity_mode == 1 or sensitivity_mode == 2)
    assert( f_sky > 0.0 and f_sky <= 1.0)
    assert( ell_max <= 2e4 )
    NTubes_LF  = 1 
    NTubes_MF  = 4 
    NTubes_UHF = 2 
    # sensitivity in uK*sqrt(s)
    # set noise to irrelevantly high value when NTubes=0
    # note that default noise levels are for 1-4-2 tube configuration
    S_LA_27  = np.array([1.0e9, 48.0, 35.0]) * np.sqrt(1./NTubes_LF)  ## converting these to per tube sensitivities
    S_LA_39  = np.array([1.0e9, 24.0, 18.0]) * np.sqrt(1./NTubes_LF)
    S_LA_93  = np.array([1.0e9,  5.4,  3.9]) * np.sqrt(4./NTubes_MF) 
    S_LA_145 = np.array([1.0e9,  6.7,  4.2]) * np.sqrt(4./NTubes_MF) 
    S_LA_225 = np.array([1.0e9, 15.0, 10.0]) * np.sqrt(2./NTubes_UHF) 
    S_LA_280 = np.array([1.e9,  36.0, 25.0]) * np.sqrt(2./NTubes_UHF)
    # 1/f polarization noise -- see Sec. 2.2 of SO science goals paper
    f_knee_pol_LA_27  = 700.
    f_knee_pol_LA_39  = 700.
    f_knee_pol_LA_93  = 700.
    f_knee_pol_LA_145 = 700.
    f_knee_pol_LA_225 = 700.
    f_knee_pol_LA_280 = 700.
    alpha_pol         = -1.4
    
    ####################################################################
    ## calculate the survey area and time
    survey_time = 5. #years
    t     = survey_time * 365.25 * 24. * 3600.    ## convert years to seconds
    t     = t * 0.2   ## retention after observing efficiency and cuts
    t     = t * 0.85  ## a kludge for the noise non-uniformity of the map edges
    A_SR  = 4. * np.pi * f_sky  ## sky areas in steradians
    A_deg =  A_SR * (180/np.pi)**2  ## sky area in square degrees
    #print("sky area: ", A_deg, "degrees^2")
    
    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(2, ell_max, 1)
    
    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_LA_27[sensitivity_mode]  / np.sqrt(t)
    W_T_39  = S_LA_39[sensitivity_mode]  / np.sqrt(t)
    W_T_93  = S_LA_93[sensitivity_mode]  / np.sqrt(t)
    W_T_145 = S_LA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_LA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_LA_280[sensitivity_mode] / np.sqrt(t)

    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the atmospheric contribution for P
    AN_P_27  = (ell / f_knee_pol_LA_27 )**alpha_pol + 1.  
    AN_P_39  = (ell / f_knee_pol_LA_39 )**alpha_pol + 1. 
    AN_P_93  = (ell / f_knee_pol_LA_93 )**alpha_pol + 1.   
    AN_P_145 = (ell / f_knee_pol_LA_145)**alpha_pol + 1.   
    AN_P_225 = (ell / f_knee_pol_LA_225)**alpha_pol + 1.   
    AN_P_280 = (ell / f_knee_pol_LA_280)**alpha_pol + 1.

    ## calculate N(ell)
    N_ell_P_27   = (W_T_27  * np.sqrt(2))**2. * A_SR * AN_P_27
    N_ell_P_39   = (W_T_39  * np.sqrt(2))**2. * A_SR * AN_P_39
    N_ell_P_93   = (W_T_93  * np.sqrt(2))**2. * A_SR * AN_P_93
    N_ell_P_145  = (W_T_145 * np.sqrt(2))**2. * A_SR * AN_P_145
    N_ell_P_225  = (W_T_225 * np.sqrt(2))**2. * A_SR * AN_P_225
    N_ell_P_280  = (W_T_280 * np.sqrt(2))**2. * A_SR * AN_P_280
    
    # include cross-correlations due to atmospheric noise
    # use correlation coefficient of r=0.9 within each dichroic pair and 0 otherwise
    r_atm = 0.9
    # different approach than for T -- need to subtract off the white noise part to get the purely atmospheric part
    # see Sec. 2.2 of the SO science goals paper
    N_ell_P_27_atm  = (W_T_27  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_27 )**alpha_pol
    N_ell_P_39_atm  = (W_T_39  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_39 )**alpha_pol
    N_ell_P_93_atm  = (W_T_93  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_93 )**alpha_pol
    N_ell_P_145_atm = (W_T_145  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_145 )**alpha_pol
    N_ell_P_225_atm = (W_T_225  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_225 )**alpha_pol
    N_ell_P_280_atm = (W_T_280  * np.sqrt(2))**2. * A_SR * (ell / f_knee_pol_LA_280 )**alpha_pol
    N_ell_P_27x39   = r_atm * np.sqrt(N_ell_P_27_atm * N_ell_P_39_atm)
    N_ell_P_93x145  = r_atm * np.sqrt(N_ell_P_93_atm * N_ell_P_145_atm)
    N_ell_P_225x280 = r_atm * np.sqrt(N_ell_P_225_atm * N_ell_P_280_atm)
        
    ## make a dictionary of noise curves for P
    N_ell_P_LA = {'ell':ell,
                   '27':N_ell_P_27,   '39':N_ell_P_39,    '27x39':N_ell_P_27x39,
                   '93':N_ell_P_93,  '145':N_ell_P_145,  '93x145':N_ell_P_93x145,
                  '225':N_ell_P_225, '280':N_ell_P_280, '225x280':N_ell_P_225x280}

    return N_ell_P_LA

class CMB:

    def __init__(self,libdir,nside,alpha=None,Acb=None,model='iso'):
        self.libdir = os.path.join(libdir, 'CMB')
        os.makedirs(self.libdir, exist_ok=True)
        self.nside = nside
        self.alpha = alpha
        self.lmax = 3 * nside - 1
        if os.path.isfile(spectra):
            self.powers = pl.load(open(spectra, 'rb'))
        else:
            self.powers = self.compute_powers()
        self.Acb = Acb
        assert model in ['iso','aniso'], "model should be 'iso' or 'aniso'"
        self.model = model
        if model == 'iso':
            assert alpha is not None, "alpha should be provided for isotropic model"
        if model == 'aniso':
            assert Acb is not None, "Acb should be provided for anisotropic model"
        if self.model == 'aniso':
            raise NotImplementedError("Anisotropic model is not implemented yet")

    
    def compute_powers(self):
        params = camb.read_ini(ini_file)
        results = camb.get_results(params)
        powers = {}
        powers['cls'] = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=True)
        powers['dls'] = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=False)
        if mpi.rank == 0:
            pl.dump(powers, open(spectra, 'wb'))
        mpi.barrier()
        return powers

    def get_power(self, dl=True):
        return self.powers['dls'] if dl else self.powers['cls']
    
    def get_lensed_spectra(self, dl=True,dtype='d'):
        powers = self.get_power(dl)['lensed_scalar']
        if dtype == 'd':
            pow = {}
            pow['tt'] = powers[:, 0]
            pow['ee'] = powers[:, 1]
            pow['bb'] = powers[:, 2]
            pow['te'] = powers[:, 3]
            return pow
        elif dtype == 'a':
            return powers
        else:
            raise ValueError("dtype should be 'd' or 'a'")
    
    def get_unlensed_spectra(self, dl=True,dtype='d'):
        powers = self.get_power(dl)['unlensed_scalar']
        if dtype == 'd':
            pow = {}
            pow['tt'] = powers[:, 0]
            pow['ee'] = powers[:, 1]
            pow['bb'] = powers[:, 2]
            pow['te'] = powers[:, 3]
            return pow
        elif dtype == 'a':
            return powers
        else:
            raise ValueError("dtype should be 'd' or 'a'")

    
    def get_cb_lensed_spectra(self,alpha=0.3,dl=True,dtype='d',new=False):
        powers = self.get_lensed_spectra(dl=dl)
        pow = {}
        pow['tt'] = powers['tt']
        pow['te'] = powers['te'] * np.cos(2*inrad(alpha)) # type: ignore
        pow['ee'] = (powers['ee'] * np.cos(inrad(2*alpha))**2) - (powers['bb'] * np.sin(inrad(2*alpha))**2) # type: ignore
        pow['bb'] = (powers['ee'] * np.sin(inrad(2*alpha))**2) + (powers['bb'] * np.cos(inrad(2*alpha))**2) # type: ignore
        pow['eb'] = 0.5 * (powers['ee'] - powers['bb']) * np.sin(inrad(4*alpha))  # type: ignore
        pow['tb'] = powers['te'] * np.sin(2*inrad(alpha)) # type: ignore
        if dtype == 'd':
            return pow
        elif dtype == 'a':
            if new:
                #TT, EE, BB, TE, EB, TB
                return np.array([pow['tt'],pow['ee'],pow['bb'],pow['te'],pow['eb'],pow['tb']])
            else:
                # TT, TE, TB, EE, EB, BB 
                return np.array([pow['tt'],pow['te'],pow['tb'],pow['ee'],pow['eb'],pow['bb']])
        else:
            raise ValueError("dtype should be 'd' or 'a'")
        
    def get_cb_lensed_QU(self,idx):
        fname = os.path.join(self.libdir, f"cmbQU_N{self.nside}_{str(self.alpha).replace('.','p')}_{idx:03d}.fits")
        if os.path.isfile(fname):
            return hp.read_map(fname,field=[0,1])
        else:
            spectra = self.get_lensed_spectra(dl=False,)
            T,E,B = hp.synalm([spectra['tt'],spectra['ee'],spectra['bb'],spectra['te']],lmax=self.lmax,new=True)
            del T
            Elm = (E * np.cos(inrad(2*self.alpha))) - (B * np.sin(inrad(2*self.alpha)))
            Blm = (E * np.sin(inrad(2*self.alpha))) + (B * np.cos(inrad(2*self.alpha)))
            QU = hp.alm2map_spin([Elm,Blm],self.nside,2,lmax=self.lmax)
            hp.write_map(fname,QU,dtype=np.float64)
            return QU

class BandpassInt:

    def __init__(self):
        self.bp = pl.load(open(bp_file, 'rb'))

    def get_profile(self,band):
        band = str(band)
        nu, bp = self.bp[band]
        return nu[nu > 0], bp[nu > 0]
    
    def plot_profiles(self):
        bands = self.bp.keys()
        plt.figure(figsize=(6,4))
        for i,b in enumerate(bands):
            nu, bp = self.get_profile(b)
            plt.plot(nu,bp,label=b)
        plt.xlabel('Frequency (GHz)')
        plt.legend()
        plt.tight_layout()

class Foreground:

    def __init__(self,libdir,nside,dust_model,sync_model,bandpass=False):
        self.libdir = os.path.join(libdir, 'Foregrounds')
        os.makedirs(self.libdir, exist_ok=True)
        self.nside = nside
        self.dust_model = dust_model
        self.sync_model = sync_model
        self.bandpass = bandpass
        self.bp_profile = BandpassInt() if bandpass else None

    def dustQU(self,band):
        if type(band) == str or type(band) == np.str_:
            band = int(band[:-1])
        name = f'dustQU_N{self.nside}_f{band}.fits' if not self.bandpass else f'dustQU_N{self.nside}_f{band}_bp.fits'
        fname = os.path.join(self.libdir, name)
        if os.path.isfile(fname):
            return hp.read_map(fname,field=[0,1])
        else:
            sky = pysm3.Sky(nside=self.nside, preset_strings=[f"d{int(self.dust_model)}"])
            if self.bandpass:
                nu, weights = self.bp_profile.get_profile(band)
                nu = nu * u.GHz
                maps = sky.get_emission(nu,weights)
            else:
                maps = sky.get_emission(band * u.GHz)
            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band*u.GHz))
            if mpi.rank == 0:
                hp.write_map(fname,maps[1:],dtype=np.float64)
            mpi.barrier()
            return maps[1:].value
    
    def syncQU(self,band):
        if type(band) == str or type(band) == np.str_:
            band = int(band[:-1])
        name = f'syncQU_N{self.nside}_f{band}.fits' if not self.bandpass else f'syncQU_N{self.nside}_f{band}_bp.fits'
        fname = os.path.join(self.libdir, name)
        if os.path.isfile(fname):
            return hp.read_map(fname,field=[0,1])
        else:
            sky = pysm3.Sky(nside=self.nside, preset_strings=[f"s{int(self.sync_model)}"])
            if self.bandpass:
                nu, weights = self.bp_profile.get_profile(band)
                nu = nu * u.GHz
                maps = sky.get_emission(nu,weights)
            else:
                maps = sky.get_emission(band * u.GHz)
            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band*u.GHz))
            if mpi.rank == 0:
                hp.write_map(fname,maps[1:],dtype=np.float64)
            mpi.barrier()
            return maps[1:].value

def rnd_alm(lmax):
    mock_cl = np.repeat(1.0e3, lmax+1)
    return hp.almxfl(hp.synalm(mock_cl, lmax=lmax, new=True), 1/np.sqrt(mock_cl))

def cholesky_matrix_elements(N_ell, lmax):
    L11     = np.zeros(lmax, dtype=float)
    L11[2:] = np.sqrt(N_ell['27']) 
    L21     = np.zeros(lmax, dtype=float)
    L21[2:] = N_ell['27x39']/np.sqrt(N_ell['27'])
    L22     = np.zeros(lmax, dtype=float)
    L22[2:] = np.sqrt((N_ell['27']*N_ell['39'] - N_ell['27x39']**2)/N_ell['27'])
    
    L33     = np.zeros(lmax, dtype=float)
    L33[2:] = np.sqrt(N_ell['93']) 
    L43     = np.zeros(lmax, dtype=float)
    L43[2:] = N_ell['93x145']/np.sqrt(N_ell['93'])
    L44     = np.zeros(lmax, dtype=float)
    L44[2:] = np.sqrt((N_ell['93']*N_ell['145'] - N_ell['93x145']**2)/N_ell['93'])
    
    L55     = np.zeros(lmax, dtype=float)
    L55[2:] = np.sqrt(N_ell['225']) 
    L65     = np.zeros(lmax, dtype=float)
    L65[2:] = N_ell['225x280']/np.sqrt(N_ell['225'])
    L66     = np.zeros(lmax, dtype=float)
    L66[2:] = np.sqrt((N_ell['225']*N_ell['280'] - N_ell['225x280']**2)/N_ell['225'])
        
    return L11, L21, L22, L33, L43, L44, L55, L65, L66


def gen_noise_maps(nside, lmax, N_ell):
    L11, L21, L22, L33, L43, L44, L55, L65, L66 = cholesky_matrix_elements(N_ell, lmax)
    
    alm     = rnd_alm(lmax)
    blm     = rnd_alm(lmax)
    nlm_27  = hp.almxfl(alm, L11)
    nlm_39  = hp.almxfl(alm, L21) + hp.almxfl(blm, L22)
    n_27    = hp.alm2map(nlm_27, nside, pixwin=False)
    n_39    = hp.alm2map(nlm_39, nside, pixwin=False)
    
    clm     = rnd_alm(lmax)
    dlm     = rnd_alm(lmax)
    nlm_93  = hp.almxfl(clm, L33)
    nlm_145 = hp.almxfl(clm, L43) + hp.almxfl(dlm, L44)
    n_93    = hp.alm2map(nlm_93,  nside, pixwin=False)
    n_145   = hp.alm2map(nlm_145, nside, pixwin=False)
    
    elm     = rnd_alm(lmax)
    flm     = rnd_alm(lmax)
    nlm_225 = hp.almxfl(elm, L55)
    nlm_280 = hp.almxfl(elm, L65) + hp.almxfl(flm, L66)
    n_225   = hp.alm2map(nlm_225, nside, pixwin=False)
    n_280   = hp.alm2map(nlm_280, nside, pixwin=False)
    
    return np.array([n_27, n_39, n_93, n_145, n_225, n_280])

class Noise:
    nlevp = np.array([71,36,8,10,22,54,71,36,8,10,22,54])

    def __init__(self,nside,atm_noise=False):
        self.nside = nside
        self.lmax = 3 * nside - 1
        self.fsky = 0.6
        self.sensitivity_mode = 2
        self.atm_noise = atm_noise
        if atm_noise:
            self.Nell = SO_LAT_Nell(self.sensitivity_mode,self.fsky,self.lmax)
            print("Noise Model: Atmospheric noise")
        else:
            print("Noise Model: White noise")

    def noiseQU(self):
        if self.atm_noise:
            N = self.noiseQUatm()
        else:
            N = self.noiseQUwhite()
        return N
    
    def noiseQUwhite(self):
        depth_p = self.nlevp
        depth_i = depth_p/np.sqrt(2)
        pix_amin2 = 4. * np.pi / float(hp.nside2npix(self.nside)) * (180. * 60. / np.pi) ** 2
        sigma_pix_I = np.sqrt(depth_i ** 2 / pix_amin2)
        sigma_pix_P = np.sqrt(depth_p ** 2 / pix_amin2)
        npix = hp.nside2npix(self.nside)
        noise = np.random.randn(len(depth_i), 3, npix)
        noise[:, 0, :] *= sigma_pix_I[:, None]
        noise[:, 1, :] *= sigma_pix_P[:, None]
        noise[:, 2, :] *= sigma_pix_P[:, None]
        return noise[:,1:, :] * np.sqrt(2)
    
    def noiseQUatm(self):
        q1 =  gen_noise_maps(self.nside, self.lmax, self.Nell)
        u1 = gen_noise_maps(self.nside, self.lmax, self.Nell)
        q2 =  gen_noise_maps(self.nside, self.lmax, self.Nell)
        u2 = gen_noise_maps(self.nside, self.lmax, self.Nell) 
        qu = []
        for i in range(len(q1)):
            qu.append([q1[i],u1[i]])
        for i in range(len(q2)):
            qu.append([q2[i],u2[i]])
        return np.array(qu) * np.sqrt(2)

class LATsky:
    freqs = np.array(['27a','39a','93a','145a','225a','280a','27b','39b','93b','145b','225b','280b'])
    fwhm = np.array([7.4,5.1,2.2,1.4,1.0,0.9,7.4,5.1,2.2,1.4,1.0,0.9])
    nlevp = np.array([71,36,8,10,22,54,71,36,8,10,22,54])
    configs = {}
    for i in range(len(freqs)):
        configs[freqs[i]] = {'fwhm':fwhm[i],'nlevp':nlevp[i]}
    
    def __init__(self,libdir,nside,alpha,dust,synch,beta,atm_noise=False,nhits=False,bandpass=False):
        fldname = ''
        if atm_noise:
            fldname += '_atm_noise'
        self.libdir = os.path.join(libdir, 'LAT'+fldname)
        os.makedirs(self.libdir, exist_ok=True)
        self.config = self.configs
        self.nside = nside
        self.alpha = alpha
        self.cmb = CMB(libdir,nside,alpha)
        self.foreground = Foreground(libdir,nside,dust,synch,bandpass)
        self.dust = dust
        self.synch = synch
        self.noise = Noise(nside,atm_noise)
        if type(beta) == list:
            assert len(beta) == len(self.freqs)
            beta_dict = {}
            for i,b in enumerate(beta):
                self.config[self.freqs[i]]['beta'] = b
        else:
            for f in self.freqs:
                self.config[f]['beta'] = beta

        self.beta = beta
        self.mask = Mask(nside,self.libdir).get_mask(nhits)
        self.atm_noise = atm_noise
        self.nhits = nhits
        if self.nhits:
            raise NotImplementedError("nhits is not implemented yet")
        self.bandpass = bandpass
        if self.bandpass:
            print("Bandpass is enabled")


    def signalOnlyQU(self,idx,band):
        if type(band) == str or type(band) == np.str_:
            band = int(band[:-1])
        cmbQU = np.array(self.cmb.get_cb_lensed_QU(idx))
        dustQU = self.foreground.dustQU(band)
        syncQU = self.foreground.syncQU(band)
        return cmbQU + dustQU + syncQU
    
    def obsQUwBeta(self,idx,band,fwhm,beta):
        signal = self.signalOnlyQU(idx,band)
        E, B = hp.map2alm_spin(signal,2,lmax=self.cmb.lmax)
        Elm = (E * np.cos(inrad(2*beta))) - (B * np.sin(inrad(2*beta)))
        Blm = (E * np.sin(inrad(2*beta))) + (B * np.cos(inrad(2*beta)))
        del (E,B)
        bl = hp.gauss_beam(np.radians(fwhm/60),lmax=self.cmb.lmax)
        hp.almxfl(Elm,bl,inplace=True)
        hp.almxfl(Blm,bl,inplace=True)
        return hp.alm2map_spin([Elm,Blm],self.nside,2,lmax=self.cmb.lmax)
    
    def obsQUfname(self,idx,band):
        fwhm = self.config[band]['fwhm']
        beta = self.config[band]['beta']
        alpha = self.cmb.alpha
        return os.path.join(self.libdir, f"obsQU_N{self.nside}_b{str(beta).replace('.','p')}_a{str(alpha).replace('.','p')}_{band}{'_bp' if self.bandpass else ''}_{fwhm}_{idx:03d}.fits")
    
    def saveObsQUs(self,idx):
        signal = []
        for band in self.freqs:
            fwhm = self.config[band]['fwhm']
            beta = self.config[band]['beta']
            signal.append(self.obsQUwBeta(idx,band,fwhm,beta))
        noise = self.noise.noiseQU()
        sky = np.array(signal) + noise
        for i in tqdm(range(len(self.freqs)),desc='Saving Observed QUs',unit='band'):
            band = self.freqs[i]
            fwhm = self.config[band]['fwhm']
            beta = self.config[band]['beta']
            fname = self.obsQUfname(idx,band)
            hp.write_map(fname,sky[i],dtype=np.float64)
        
        
    def obsQU(self,idx,band):
        fwhm = self.config[band]['fwhm']
        beta = self.config[band]['beta']
        fname = self.obsQUfname(idx,band)
        if os.path.isfile(fname):
            return hp.read_map(fname,field=[0,1])
        else:
            self.saveObsQUs(idx)
            return hp.read_map(fname,field=[0,1])


class Mask:
    def __init__(self,nside,libdir=None,) -> None:
        self.nside = nside
        self.libdir = libdir
        self.mask_save = False if libdir is None else True
        self.fsky = np.mean(self.get_mask(nhits=False))

    def get_mask(self,nhits=False):
        ivar = hp.read_map(mask)
        if self.nside != hp.get_nside(ivar):
            ivar = hp.ud_grade(ivar,self.nside)
        if nhits:
            return ivar
        else:
            return (ivar > 0).astype(int)


    def get_mask_deprecated(self):
        fname = os.path.join(self.libdir, f"mask_N{self.nside}.fits")
        if os.path.isfile(fname):
            return hp.read_map(fname)
        mask_percent_n = 35
        mask_percent_s = 5
        npix = hp.nside2npix(self.nside)
        mask = np.zeros(npix)
        from_i = int(npix*mask_percent_n/100)
        to_i = npix - int(npix*mask_percent_s/100)
        mask[from_i:to_i] = 1
        mask = self.change_coord(mask, ['C', 'G'])
        if self.mask_save and (mpi.rank == 0):
            hp.write_map(fname, mask,dtype=np.int32)
        return mask

    @staticmethod
    def change_coord(m, coord):
        npix = m.shape[-1]
        nside = hp.npix2nside(npix)
        ang = hp.pix2ang(nside, np.arange(npix))
        rot = hp.Rotator(coord=reversed(coord))
        new_ang = rot(*ang)
        new_pix = hp.ang2pix(nside, *new_ang)
        return m[..., new_pix]


