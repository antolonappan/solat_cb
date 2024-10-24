import numpy as np
from tqdm import tqdm
from dataclasses import dataclass,field
from typing import Any, Optional
from solat_cb.simulation import CMB
import os
import healpy as hp
import pickle as pl
import emcee
from getdist import plots, MCSamples


def selector(lib,lmin,lmax,):
    b = lib.binInfo.get_effective_ells()
    select = np.where((b>lmin) & (b<lmax))[0]
    return select

def get_sp(lib,lmax):
    bl_arr = []
    for i in range(6):
        bl_arr.append(lib.binInfo.bin_cell(hp.gauss_beam(np.radians(lib.lat.fwhm[i]/60),lmax)**2))
    bl_arr = np.array(bl_arr)
    obs_arr = []
    for i in range(20):
        sp = lib.obs_x_obs(i)
        obs_arr.append(sp)
    obs_arr = np.array(obs_arr)
    obs_arr = obs_arr[:,np.arange(6),np.arange(6),2,2:]
    return obs_arr/bl_arr

def paranames(lib,name):
    return [f"a{name}{fe}" for fe in lib.lat.freqs]
def latexnames(lib, name):
    return [r'\alpha_{{{}}}^{{{}}}'.format(name, fe) for fe in lib.lat.freqs]

class Sat4Lat:
    
    def __init__(self,libdir,latlib,satlib,lmin,lmax,sat_err,alpha_sat,alpha_lat,beta):
        self.libdir = os.path.join(libdir,'Calibration')
        os.makedirs(self.libdir,exist_ok=True)
        self.sat_err = sat_err
        self.binner = latlib.binInfo
        self.Lmax = latlib.lmax
        self.__select__ = selector(latlib,lmin,lmax)
        self.lat_mean, self.lat_std = self.calc_mean_std(latlib)
        self.sat_mean, self.sat_std = self.calc_mean_std(satlib)
        self.cl_len = CMB(libdir,latlib.lat.nside,).get_lensed_spectra(dl=False)
        self.true =  np.concatenate([alpha_lat,alpha_sat,[beta]])
        self.__asat__ = alpha_sat
        self.__alat__ = alpha_lat
        self.__pnames__ = paranames(latlib,'LAT') + paranames(satlib,'SAT') + ['beta']
        self.__plabels__ = latexnames(latlib,'LAT') + latexnames(satlib,'SAT') + [r'\beta']
    
    def calc_mean_std(self,lib):
        sp = get_sp(lib,self.Lmax)
        return ( sp.mean(axis=0)[:,self.__select__],
                 sp.std(axis=0)[:,self.__select__] )
    
    def theory(self,beta_array):
        beta_array = np.asarray(beta_array)
        th = 0.5 * (self.cl_len["ee"] - self.cl_len["bb"])[:, np.newaxis] * np.sin(np.deg2rad(4 * beta_array))
        return np.apply_along_axis(lambda th_slice: self.binner.bin_cell(th_slice[:self.Lmax+1])[self.__select__], 0, th).T
    
    def chisq(self,theta):
        alpha_lat,alpha_sat,beta = np.array(theta[:6]), np.array(theta[6:12]), np.ones(6)*theta[-1]

        diff_mean = self.lat_mean - self.sat_mean
        diff_std = np.sqrt(self.lat_std**2 + self.sat_std**2)  
        diff_model = self.theory(alpha_lat-alpha_sat)
        diff_chi = np.sum(((diff_mean - diff_model)/diff_std)**2)

        sat_model = self.theory(beta + alpha_sat)
        sat_chi = np.sum(((self.sat_mean - sat_model)/self.sat_std)**2)

        lat_model = self.theory(beta + alpha_lat)
        lat_chi = np.sum(((self.lat_mean - lat_model)/self.lat_std)**2)

        return diff_chi + sat_chi + lat_chi
    
    def lnprior(self,theta):
        alphalat,alphasat,beta = np.array(theta[:6]), np.array(theta[6:12]), theta[-1]
        sigma = self.sat_err
        lnp = -0.5 * (alphasat - self.__asat__)**2 / sigma**2 - np.log(sigma*np.sqrt(2*np.pi))
        if np.all(alphalat > -0.5) and np.all(alphalat < 0.5) and -0.1 < beta < 0.5:
            return np.sum(lnp)
        return -np.inf


    def ln_likelihood(self,theta):
        return -0.5 * self.chisq(theta)

    def ln_prob(self,theta):
        lp = self.lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.ln_likelihood(theta)
    
    
    def samples(self,nwalkers=32,nsamples=1000):
        fname = os.path.join(self.libdir,f'samples_w{nwalkers}_n{nsamples}.pkl')
        if os.path.isfile(fname):
            flat_samples = pl.load(open(fname,'rb'))
        else:
            ndim = len(self.true)
            pos = self.true + 1e-3 * np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.ln_prob, threads=4)
            sample = sampler.run_mcmc(pos, nsamples, progress=True)
            flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
            pl.dump(flat_samples,open(fname,'wb'))
        return flat_samples
    
    def getdist_samples(self,nwalkers,nsamples):
        flat_samples = self.samples(nwalkers,nsamples)
        return MCSamples(samples=flat_samples,names = self.__pnames__, labels = self.__plabels__)
        
    
    def plot_getdist(self,nwalkers,nsamples,avoid_sat=False,beta_only=False):
        flat_samples = self.getdist_samples(nwalkers,nsamples)
        if beta_only:
            g = plots.get_single_plotter(width_inch=4)
            g.plot_1d(flat_samples, 'beta', title_limit=1)
        else:
            names = self.__pnames__
            if avoid_sat:
                names = [item for item in names if 'SAT' not in item]
            g = plots.get_subplot_plotter()
            g.triangle_plot([flat_samples], names, filled=True,title_limit=1)

