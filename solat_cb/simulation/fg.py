"""
This file contains the implementation of the Foreground class for generating and handling dust and synchrotron foreground maps.
"""
# General imports
import os
import pysm3
import numpy as np
import healpy as hp
import pickle as pl
from typing import Tuple
from pysm3 import units as u
import matplotlib.pyplot as plt
# Local imports
from solat_cb import mpi
from solat_cb.utils import Logger
from solat_cb.data import BP_PROFILE


class BandpassInt:
    def __init__(
        self,
        libdir: str,
    ):
        """
        Initializes the BandpassInt class, loading bandpass profiles from a specified file.
        """
        BP_PROFILE.directory = libdir
        self.bp = BP_PROFILE.data

    def get_profile(self, band: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the frequency and bandpass profile for a specified band.

        Parameters:
        band (str): The frequency band for which to retrieve the profile.

        Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two NumPy arrays:
                                       - nu: Array of frequencies (GHz) where the bandpass is defined.
                                       - bp: Array of bandpass values corresponding to the frequencies.
        """
        nu, bp = self.bp[band]
        return nu[nu > 0], bp[nu > 0]

    def plot_profiles(self) -> None:
        """
        Plots the bandpass profiles for all available bands.

        The method iterates over all bands, retrieves the bandpass profile,
        and plots the profile as a function of frequency (GHz).
        """
        bands = self.bp.keys()
        plt.figure(figsize=(6, 4))
        for b in bands:
            nu, bp = self.get_profile(b)
            plt.plot(nu, bp, label=b)
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Bandpass Response")
        plt.legend(title="Bands")
        plt.tight_layout()
        plt.show()

class Foreground:
    def __init__(
        self,
        libdir: str,
        nside: int,
        dust_model: int,
        sync_model: int,
        bandpass: bool = False,
        verbose: bool = True,
    ):
        """
        Initializes the Foreground class for generating and handling dust and synchrotron foreground maps.

        Parameters:
        libdir (str): Directory where the foreground maps will be stored.
        nside (int): HEALPix resolution parameter.
        dust_model (int): Model number for the dust emission.
        sync_model (int): Model number for the synchrotron emission.
        bandpass (bool, optional): If True, bandpass integration is applied. Defaults to False.
        """
        self.logger = Logger(self.__class__.__name__, verbose=verbose)
        self.libdir = os.path.join(libdir, "Foregrounds")
        if mpi.rank == 0:
            os.makedirs(self.libdir, exist_ok=True)
        mpi.barrier()
        self.nside = nside
        self.dust_model = dust_model
        self.sync_model = sync_model
        self.bandpass = bandpass
        if bandpass:
            self.bp_profile = BandpassInt(libdir)
            self.logger.log("Bandpass integration is enabled", level="info")
        else:
            self.bp_profile = None
            self.logger.log("Bandpass integration is disabled", level="info")

    def dustQU(self, band: str) -> np.ndarray:
        """
        Generates or retrieves the Q and U Stokes parameters for dust emission at a given frequency band.

        Parameters:
        band (str): The frequency band.

        Returns:
        np.ndarray: A NumPy array containing the Q and U maps.
        """
        name = (
            f"dustQU_N{self.nside}_f{band}.fits"
            if not self.bandpass
            else f"dustQU_N{self.nside}_f{band}_bp.fits"
        )
        fname = os.path.join(self.libdir, name)

        if os.path.isfile(fname):
            self.logger.log(f"Loading dust Q and U maps for band {band}", level="info")
            return hp.read_map(fname, field=[0, 1]) # type: ignore
        
        else:
            self.logger.log(f"Generating dust Q and U maps for band {band}", level="info")
            sky = pysm3.Sky(
                nside=self.nside, preset_strings=[f"d{int(self.dust_model)}"]
            )
            if self.bandpass:
                if self.bp_profile is not None:
                    nu, weights = self.bp_profile.get_profile(band)
                else:
                    raise ValueError("Bandpass profile is not initialized.")
                nu = nu * u.GHz # type: ignore
                maps = sky.get_emission(nu, weights)
            else:
                maps = sky.get_emission(int(band) * u.GHz) # type: ignore

            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(int(band) * u.GHz)) # type: ignore

            if mpi.rank == 0:
                hp.write_map(fname, maps[1:], dtype=np.float64) # type: ignore
            mpi.barrier()

            return maps[1:].value

    def syncQU(self, band: str) -> np.ndarray:
        """
        Generates or retrieves the Q and U Stokes parameters for synchrotron emission at a given frequency band.

        Parameters:
        band (str): The frequency band.

        Returns:
        np.ndarray: A NumPy array containing the Q and U maps.
        """
        name = (
            f"syncQU_N{self.nside}_f{band}.fits"
            if not self.bandpass
            else f"syncQU_N{self.nside}_f{band}_bp.fits"
        )
        fname = os.path.join(self.libdir, name)

        if os.path.isfile(fname):
            self.logger.log(f"Loading synchrotron Q and U maps for band {band}", level="info")
            return hp.read_map(fname, field=[0, 1]) # type: ignore
        else:
            self.logger.log(f"Generating synchrotron Q and U maps for band {band}", level="info")
            sky = pysm3.Sky(
                nside=self.nside, preset_strings=[f"s{int(self.sync_model)}"]
            )
            if self.bandpass:
                if self.bp_profile is not None:
                    nu, weights = self.bp_profile.get_profile(band)
                else:
                    raise ValueError("Bandpass profile is not initialized.")
                nu = nu * u.GHz # type: ignore
                maps = sky.get_emission(nu, weights)
            else:
                maps = sky.get_emission(int(band) * u.GHz) # type: ignore

            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(int(band) * u.GHz)) # type: ignore

            if mpi.rank == 0:
                hp.write_map(fname, maps[1:], dtype=np.float64) # type: ignore
            mpi.barrier()

            return maps[1:].value