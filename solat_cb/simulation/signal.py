import numpy as np
import healpy as hp
import pysm3
from pysm3 import units as u
import camb
import os
import pickle as pl
from tqdm import tqdm
from solat_cb import mpi
import matplotlib.pyplot as plt  
from typing import Dict, Optional, Any, Union, List, Tuple
import requests






#TODO at the moment doesn't support anisotropic birefringence
class LATsky:
    # Understanding data splits as maps made by coadding the data from different
    # fractions of the total observation time (e.g., first and second half), i.e.:
    # - white noise is independent between the frequencies within one split, and
    # between different splits
    # - 1/f noise can be correlated between frequencies of the same split, but will 
    # be independent between different splits
    # - for now, assume detectors are unchanged through observation campaings and
    # polarisation angles for each frequency are the same across different splits

    freqs = np.array(["27","39","93","145","225","280"])
    fwhm  = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9]) # arcmin
    tube  = np.array(["LF", "LF", "MF", "MF", "HF", "HF"]) # tube each frequency occupies

    def __init__(
        self,
        libdir: str,
        nside: int,
        beta: float,
        dust: int,
        synch: int,
        alpha: Union[float, List[float]],
        atm_noise: bool = False,
        nsplits: int = 2,
        bandpass: bool = False,
    ):
        """
        Initializes the LATsky class for generating and handling Large Aperture Telescope (LAT) sky simulations.

        Parameters:
        libdir (str): Directory where the sky maps will be stored.
        nside (int): HEALPix resolution parameter.
        beta (float): Rotation angle for cosmic birefringence in degrees.
        dust (int): Model number for the dust emission.
        synch (int): Model number for the synchrotron emission.
        alpha (Union[float, List[float]]): polarisation angle(s) for frequency bands. If a list, should match the number of frequency bands.
        atm_noise (bool, optional): If True, includes atmospheric noise. Defaults to False.
        nhits (bool, optional): If True, includes hit count map. Defaults to False.
        nsplits (int, optional): Number of data splits to consider. Defaults to 2.
        bandpass (bool, optional): If True, applies bandpass integration. Defaults to False.
        """
        fldname     = "_atm_noise" if atm_noise else "_white_noise"
        fldname    += f"_{nsplits}splits"
        self.libdir = os.path.join(libdir, "LAT" + fldname)
        os.makedirs(self.libdir+'/obs', exist_ok=True)

        
        self.config = {}
        for split in range(nsplits):
            for band in range(len(self.freqs)):
                self.config[f'{self.freqs[band]}-{split+1}'] = {"fwhm": self.fwhm[band], "opt. tube": self.tube[band]}
        self.nside      = nside
        self.beta       = beta
        self.cmb        = CMB(libdir, nside, beta)
        self.foreground = Foreground(libdir, nside, dust, synch, bandpass)
        self.dust_model = dust
        self.sync_model = synch
        self.nsplits    = nsplits
        self.mask, self.fsky = self.__set_mask_fsky__(libdir)
        self.noise      = Noise(nside, self.fsky, atm_noise, nsplits)

        if isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == len(
                self.freqs
            ), "Length of alpha list must match the number of frequency bands."
            for band, a in enumerate(alpha):
                for split in range(nsplits):
                    self.config[f'{self.freqs[band]}-{split+1}']["alpha"] = a
        else:
            for split in range(nsplits):
                for band in range(len(self.freqs)):
                    self.config[f'{self.freqs[band]}-{split+1}']["alpha"] = alpha

        self.alpha     = alpha
        self.atm_noise = atm_noise
        self.bandpass  = bandpass
        if bandpass:
            print("Bandpass is enabled")

    
    def __set_mask_fsky__(self,libdir):
        maskobj = Mask(libdir,self.nside,'LAT')
        return maskobj.mask, maskobj.fsky

    def signalOnlyQU(self, idx: int, band: str) -> np.ndarray:
        """
        Generates the Q and U Stokes parameters for the given frequency band, combining CMB, dust, and synchrotron signals.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        band (str): The frequency band.

        Returns:
        np.ndarray: A NumPy array containing the combined Q and U maps.
        """
        band   = band[:band.index('-')]
        cmbQU  = np.array(self.cmb.get_cb_lensed_QU(idx))
        dustQU = self.foreground.dustQU(band)
        syncQU = self.foreground.syncQU(band)
        return cmbQU + dustQU + syncQU

    def obsQUwAlpha(
        self, idx: int, band: str, fwhm: float, alpha: float
    ) -> np.ndarray:
        """
        Generates the observed Q and U Stokes parameters after applying a rotation by alpha and smoothing with the pixel window function a Gaussian beam.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        band (Union[str, int]): The frequency band. Can be an integer or a string (e.g., '93GHz').
        fwhm (float): Full-width half-maximum of the Gaussian beam in arcminutes.
        alpha (float): Polarisation angle in degrees.

        Returns:
        np.ndarray: A NumPy array containing the observed Q and U maps.
        """
        signal = self.signalOnlyQU(idx, band)
        E, B   = hp.map2alm_spin(signal, 2, lmax=self.cmb.lmax)
        Elm    = (E * np.cos(inrad(2 * alpha))) - (B * np.sin(inrad(2 * alpha)))
        Blm    = (E * np.sin(inrad(2 * alpha))) + (B * np.cos(inrad(2 * alpha)))
        del (E, B)
        bl     = hp.gauss_beam(inrad(fwhm / 60), lmax=self.cmb.lmax, pol=True)
        pwf    = np.array(hp.pixwin(self.nside, pol=True, lmax=self.cmb.lmax))
        hp.almxfl(Elm, bl[:,1]*pwf[1,:], inplace=True)
        hp.almxfl(Blm, bl[:,2]*pwf[1,:], inplace=True)
        return hp.alm2map_spin([Elm, Blm], self.nside, 2, lmax=self.cmb.lmax)


    def obsQUfname(self, idx: int, band: str) -> str:
        """
        Generates the filename for the observed Q and U Stokes parameter maps.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        band (str): The frequency band identifier (e.g., '93a').

        Returns:
        str: The file path for the observed Q and U maps.
        """
        alpha = self.config[band]["alpha"]
        beta  = self.cmb.beta
        return os.path.join(
            self.libdir,
            f"obs/obsQU_N{self.nside}_b{str(beta).replace('.','p')}_a{str(alpha).replace('.','p')}_{band}{'_bp' if self.bandpass else ''}_{idx:03d}.fits",
        )
    
    def saveObsQUs(self, idx: int) -> None:
        """
        Saves the observed Q and U Stokes parameter maps for all frequency bands.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        """
        bands  = list(self.config.keys())
        signal = []
        for band in bands:
            fwhm  = self.config[band]["fwhm"]
            alpha = self.config[band]["alpha"]
            signal.append(self.obsQUwAlpha(idx, band, fwhm, alpha))
        noise = self.noise.noiseQU()
        sky   = np.array(signal) + noise
        for i in tqdm(range(len(bands)), desc="Saving Observed QUs", unit="band"):
            fname = self.obsQUfname(idx, bands[i])
            hp.write_map(fname, sky[i]*self.mask, dtype=np.float64, overwrite=True)

    def obsQU(self, idx: int, band: str) -> np.ndarray:
        """
        Retrieves the observed Q and U Stokes parameter maps for a specific frequency band.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        band (str): The frequency band identifier (e.g., '93a').

        Returns:
        np.ndarray: A NumPy array containing the observed Q and U maps.
        """
        fname = self.obsQUfname(idx, band)
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])
        else:
            self.saveObsQUs(idx)
            return hp.read_map(fname, field=[0, 1])

