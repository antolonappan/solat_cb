import numpy as np
import healpy as hp
import pysm3
from pysm3 import units as u
import camb
import os
import pickle as pl
from tqdm import tqdm

ini_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cb.ini")
spectra = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spectra.pkl")
mask = os.path.join(os.path.dirname(os.path.realpath(__file__)), "masks.fits")

def inrad(alpha):
    return np.deg2rad(alpha) # type: ignore

class CMB:

    def __init__(self,libdir,nside,alpha):
        self.libdir = os.path.join(libdir, 'CMB')
        os.makedirs(self.libdir, exist_ok=True)
        self.nside = nside
        self.alpha = alpha
        self.lmax = 3 * nside - 1
        if os.path.isfile(spectra):
            self.powers = pl.load(open(spectra, 'rb'))
        else:
            self.powers = self.compute_powers()
    
    def compute_powers(self):
        params = camb.read_ini(ini_file)
        results = camb.get_results(params)
        powers = {}
        powers['cls'] = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=True)
        powers['dls'] = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=False)
        pl.dump(powers, open(spectra, 'wb'))
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

class Foreground:

    def __init__(self,libdir,nside,dust_model,sync_model):
        self.libdir = os.path.join(libdir, 'Foregrounds')
        os.makedirs(self.libdir, exist_ok=True)
        self.nside = nside
        self.dust_model = dust_model
        self.sync_model = sync_model

    def dustQU(self,band):
        fname = os.path.join(self.libdir, f'dustQU_N{self.nside}_f{band}.fits')
        if os.path.isfile(fname):
            return hp.read_map(fname,field=[0,1])
        else:
            sky = pysm3.Sky(nside=self.nside, preset_strings=[f"d{int(self.dust_model)}"])
            maps = sky.get_emission(band * u.GHz)
            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band*u.GHz))
            hp.write_map(fname,maps[1:],dtype=np.float32)
            return maps[1:].value
    
    def syncQU(self,band):
        fname = os.path.join(self.libdir, f'syncQU_N{self.nside}_{band}.fits')
        if os.path.isfile(fname):
            return hp.read_map(fname,field=[0,1])
        else:
            sky = pysm3.Sky(nside=self.nside, preset_strings=[f"s{int(self.sync_model)}"])
            maps = sky.get_emission(band * u.GHz)
            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band*u.GHz))
            hp.write_map(fname,maps[1:],dtype=np.float32)
            return maps[1:].value

class Noise:

    def __init__(self,nside):
        self.nside = nside
    
    def noiseQU(self,nlevp):
        if type(nlevp) == np.int64 or type(nlevp) == np.float:
            nlevp = [nlevp]
            depth_p = np.array(nlevp)
            only_one = True
        elif type(nlevp) == list:
            depth_p = np.array(nlevp)
            only_one = False
        elif type(nlevp) == np.ndarray:
            depth_p = nlevp
            only_one = False
        elif type(nlevp) != list:
            raise ValueError("nlevp should be a list or a number")
        
        depth_i = depth_p/np.sqrt(2)
        pix_amin2 = 4. * np.pi / float(hp.nside2npix(self.nside)) * (180. * 60. / np.pi) ** 2
        sigma_pix_I = np.sqrt(depth_i ** 2 / pix_amin2)
        sigma_pix_P = np.sqrt(depth_p ** 2 / pix_amin2)
        npix = hp.nside2npix(self.nside)
        noise = np.random.randn(len(depth_i), 3, npix)
        noise[:, 0, :] *= sigma_pix_I[:, None]
        noise[:, 1, :] *= sigma_pix_P[:, None]
        noise[:, 2, :] *= sigma_pix_P[:, None]
        if only_one:
            return noise[0][1:]
        else:
            return noise[:,1:, :]


class LATsky:
    freqs = np.array([27,39,93,145,225,280])
    fwhm = np.array([7.4,5.1,2.2,1.4,1.0,0.9])
    nlevp = np.array([71,36,8,10,22,54])
    configs = {}
    for i in range(len(freqs)):
        configs[freqs[i]] = {'fwhm':fwhm[i],'nlevp':nlevp[i]}
    
    def __init__(self,libdir,nside,alpha,dust,synch,beta):
        self.libdir = os.path.join(libdir, 'LAT')
        os.makedirs(self.libdir, exist_ok=True)
        self.config = self.configs
        self.nside = nside
        self.cmb = CMB(libdir,nside,alpha)
        self.foreground = Foreground(libdir,nside,dust,synch)
        self.noise = Noise(nside)
        if type(beta) == list:
            assert len(beta) == len(self.freqs)
            beta_dict = {}
            for i,b in enumerate(beta):
                self.config[self.freqs[i]]['beta'] = b
        else:
            for f in self.freqs:
                self.config[f]['beta'] = beta

        self.beta = beta
        self.mask = Mask(nside,self.libdir).get_mask()


    def signalQU(self,idx,band):
        cmbQU = np.array(self.cmb.get_cb_lensed_QU(idx))
        dustQU = self.foreground.dustQU(band)
        syncQU = self.foreground.syncQU(band)
        return cmbQU + dustQU + syncQU
    
    def __obsQU__(self,idx,band,fwhm,beta):
        signal = self.signalQU(idx,band)
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
        return os.path.join(self.libdir, f"obsQU_N{self.nside}_b{str(beta).replace('.','p')}_a{str(alpha).replace('.','p')}_{band}_{fwhm}_{idx:03d}.fits")
    
    def obsQU(self,idx,band):
        fwhm = self.config[band]['fwhm']
        beta = self.config[band]['beta']
        fname = self.obsQUfname(idx,band)
        if os.path.isfile(fname):
            return hp.read_map(fname,field=[0,1])
        else:
            signal = self.__obsQU__(idx,band,fwhm,beta)
            noise = self.noise.noiseQU(self.config[band]['nlevp'])
            sky = signal + noise
            hp.write_map(fname,sky,dtype=np.float64)
            return sky
    
    def obsQUs(self,idx):
        obs = []
        for f in self.freqs:
            obs.append(self.obsQU(idx,f))
        return np.array(obs)

        
        

class Mask:
    def __init__(self,nside,libdir=None,) -> None:
        self.nside = nside
        self.libdir = libdir
        self.mask_save = False if libdir is None else True

    def get_mask(self):
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
        if self.mask_save:
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


