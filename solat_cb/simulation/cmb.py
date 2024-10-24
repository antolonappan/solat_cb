"""
This file contains the class to handle the Cosmic Microwave Background (CMB) data and simulations.
"""

# General imports
import os
import camb
import numpy as np
import pickle as pl
import healpy as hp
from typing import Dict, Optional, Any, Union, List
# Local imports
from solat_cb import mpi
from solat_cb.utils import Logger, inrad
from solat_cb.data import CAMB_INI, SPECTRA

class CMB:

    def __init__(
        self,
        libdir: str,
        nside: int,
        beta: Optional[float]=None,
        Acb: Optional[float]=None,
        model: str = "iso",
        verbose: bool = True,
    ):
        self.logger = Logger(self.__class__.__name__, verbose=verbose)
        self.basedir = libdir
        self.libdir = os.path.join(libdir, "CMB")
        os.makedirs(self.libdir, exist_ok=True)
        self.nside  = nside
        self.beta   = beta
        self.lmax   = 3 * nside - 1
        SPECTRA.directory = self.basedir
        self.__spectra_file__ = SPECTRA.fname
        if os.path.isfile(self.__spectra_file__):
            self.logger.log("Loading CMB power spectra from file", level="info")
            self.powers = pl.load(open(self.__spectra_file__, "rb"))
        else:
            self.powers = SPECTRA.data
            lmax_infile = len(self.powers['cls']['lensed_scalar'][:, 0])
            if lmax_infile < self.lmax:
                self.logger.log("CMB power spectra file does not contain enough data", level="warning")
                self.logger.log("Computing CMB power spectra", level="info")
                self.powers = self.compute_powers()
                #TODO: feed the lmax to the compute_powers method
                self.logger.log("CMB power spectra computed doesn't guarantee the lmax", level="critical")
            else:
                self.logger.log("CMB power spectra file is up-to-date", level="info")
        self.Acb    = Acb
        assert model in ["iso", "aniso"], "model should be 'iso' or 'aniso'"
        self.model  = model
        if self.model == "aniso":
            self.logger.log("Anisotropic cosmic birefringence model selected", level="info")
        if self.model == "iso":
            self.logger.log("Isotropic cosmic birefringence model selected", level="info")   

    def compute_powers(self) -> Dict[str, Any]:
        """
        compute the CMB power spectra using CAMB.
        """
        CAMB_INI.directory = self.basedir
        params   = CAMB_INI.data
        results  = camb.get_results(params)
        powers   = {}
        powers["cls"] = results.get_cmb_power_spectra(
            params, CMB_unit="muK", raw_cl=True
        )
        powers["dls"] = results.get_cmb_power_spectra(
            params, CMB_unit="muK", raw_cl=False
        )
        if mpi.rank == 0:
            pl.dump(powers, open(self.__spectra_file__, "wb"))
        mpi.barrier()
        return powers

    def get_power(self, dl: bool = True) -> Dict[str, np.ndarray]:
        """
        Get the CMB power spectra.

        Parameters:
        dl (bool): If True, return the power spectra with dl factor else without dl factor.

        Returns:
        Dict[str, np.ndarray]: A dictionary containing the CMB power spectra.
        """
        return self.powers["dls"] if dl else self.powers["cls"]

    def get_lensed_spectra(
        self, dl: bool = True, dtype: str = "d"
    ) -> Union[Dict[str, Any], np.ndarray]:
        """
        Retrieve the lensed scalar spectra from the power spectrum data.

        Parameters:
        dl (bool, optional): If True, returns Dl (C_l * l * (l + 1) / 2π). Defaults to True.
        dtype (str, optional): Specifies the format of the returned spectra.
                               - 'd' returns a dictionary with keys 'tt', 'ee', 'bb', 'te'.
                               - 'a' returns the full array of power spectra.
                               Defaults to 'd'.

        Returns:
        Union[Dict[str, Any], np.ndarray]:
            - A dictionary containing individual power spectra for 'tt', 'ee', 'bb', 'te' if dtype is 'd'.
            - The full array of lensed scalar power spectra if dtype is 'a'.

        Raises:
        ValueError: If `dtype` is not 'd' or 'a'.
        """
        powers = self.get_power(dl)["lensed_scalar"]
        if dtype == "d":
            pow = {}
            pow["tt"] = powers[:, 0]
            pow["ee"] = powers[:, 1]
            pow["bb"] = powers[:, 2]
            pow["te"] = powers[:, 3]
            return pow
        elif dtype == "a":
            return powers
        else:
            raise ValueError("dtype should be 'd' or 'a'")

    def get_unlensed_spectra(
        self, dl: bool = True, dtype: str = "d"
    ) -> Union[Dict[str, Any], np.ndarray]:
        """
        Retrieve the unlensed scalar spectra from the power spectrum data.

        Parameters:
        dl (bool, optional): If True, returns Dl (C_l * l * (l + 1) / 2π). Defaults to True.
        dtype (str, optional): Specifies the format of the returned spectra.
                               - 'd' returns a dictionary with keys 'tt', 'ee', 'bb', 'te'.
                               - 'a' returns the full array of power spectra.
                               Defaults to 'd'.

        Returns:
        Union[Dict[str, Any], np.ndarray]:
            - A dictionary containing individual power spectra for 'tt', 'ee', 'bb', 'te' if dtype is 'd'.
            - The full array of unlensed scalar power spectra if dtype is 'a'.

        Raises:
        ValueError: If `dtype` is not 'd' or 'a'.
        """
        powers = self.get_power(dl)["unlensed_scalar"]
        if dtype == "d":
            pow = {}
            pow["tt"] = powers[:, 0]
            pow["ee"] = powers[:, 1]
            pow["bb"] = powers[:, 2]
            pow["te"] = powers[:, 3]
            return pow
        elif dtype == "a":
            return powers
        else:
            raise ValueError("dtype should be 'd' or 'a'")
    

    def get_cb_lensed_spectra(
        self, beta: float = 0.0, dl: bool = True, dtype: str = "d", new: bool = False
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """
        Calculate the cosmic birefringence (CB) lensed spectra with a given rotation angle `beta`.

        Parameters:
        beta (float, optional): The rotation angle in degrees for the cosmic birefringence effect. Defaults to 0.3 degrees.
        dl (bool, optional): If True, returns Dl (C_l * l * (l + 1) / 2π). Defaults to True.
        dtype (str, optional): Specifies the format of the returned spectra.
                               - 'd' returns a dictionary with keys 'tt', 'ee', 'bb', 'te', 'eb', 'tb'.
                               - 'a' returns an array of power spectra.
                               Defaults to 'd'.
        new (bool, optional): Determines the ordering of the spectra in the array if dtype is 'a'.
                              If True, returns [TT, EE, BB, TE, EB, TB].
                              If False, returns [TT, TE, TB, EE, EB, BB].
                              Defaults to False.

        Returns:
        Union[Dict[str, np.ndarray], np.ndarray]:
            - A dictionary containing individual CB lensed power spectra for 'tt', 'ee', 'bb', 'te', 'eb', 'tb' if dtype is 'd'.
            - An array of CB lensed power spectra with ordering determined by the `new` parameter if dtype is 'a'.

        Raises:
        ValueError: If `dtype` is not 'd' or 'a'.

        Notes:
        The method applies a rotation by `alpha` degrees to the E and B mode spectra to account for cosmic birefringence.
        """

        powers = self.get_lensed_spectra(dl=dl) 
        pow = {}
        pow["tt"] = powers["tt"]
        pow["te"] = powers["te"] * np.cos(2 * inrad(beta))  # type: ignore
        pow["ee"] = (powers["ee"] * np.cos(inrad(2 * beta)) ** 2) + (powers["bb"] * np.sin(inrad(2 * beta)) ** 2)  # type: ignore
        pow["bb"] = (powers["ee"] * np.sin(inrad(2 * beta)) ** 2) + (powers["bb"] * np.cos(inrad(2 * beta)) ** 2)  # type: ignore
        pow["eb"] = 0.5 * (powers["ee"] - powers["bb"]) * np.sin(inrad(4 * beta))  # type: ignore
        pow["tb"] = powers["te"] * np.sin(2 * inrad(beta))  # type: ignore
        if dtype == "d":
            return pow
        elif dtype == "a":
            if new:
                # TT, EE, BB, TE, EB, TB
                return np.array(
                    [pow["tt"], pow["ee"], pow["bb"], pow["te"], pow["eb"], pow["tb"]]
                )
            else:
                # TT, TE, TB, EE, EB, BB
                return np.array(
                    [pow["tt"], pow["te"], pow["tb"], pow["ee"], pow["eb"], pow["bb"]]
                )
        else:
            raise ValueError("dtype should be 'd' or 'a'")
    
    def get_cb_lensed_QU(self,idx: int) -> List[np.ndarray]:
        if self.model == "iso":
            return self.get_iso_cb_lensed_QU(idx)
        elif self.model == "aniso":
            return self.get_aniso_cb_lensed_QU(idx)
        else:
            raise NotImplementedError("Model not implemented yet, only 'iso' and 'aniso' are supported")
    

    def get_iso_cb_lensed_QU(self, idx: int) -> List[np.ndarray]:
        """
        Generate or retrieve the Q and U Stokes parameters after applying cosmic birefringence.

        Parameters:
        idx (int): Index for the realization of the CMB map.

        Returns:
        List[np.ndarray]: A list containing the Q and U Stokes parameter maps as NumPy arrays.

        Notes:
        The method applies a rotation to the E and B mode spherical harmonics to simulate the effect of cosmic birefringence.
        If the map for the given `idx` exists in the specified directory, it reads the map from the file.
        Otherwise, it generates the Q and U maps, applies the birefringence, and saves the resulting map to a FITS file.
        """
        fname = os.path.join(
            self.libdir,
            f"cmbQU_N{self.nside}_{str(self.beta).replace('.','p')}_{idx:03d}.fits",
        )
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])   # type: ignore
        else:
            spectra = self.get_cb_lensed_spectra(
                beta=self.beta if self.beta is not None else 0.0,
                dl=False,
            )
            # PDP: spectra start at ell=0, we are fine
            T, E, B = hp.synalm(
                [spectra["tt"], spectra["ee"], spectra["bb"], spectra["te"], spectra["eb"], spectra["tb"]],
                lmax=self.lmax,
                new=True,
            )
            del T
            QU = hp.alm2map_spin([E, B], self.nside, 2, lmax=self.lmax)
            hp.write_map(fname, QU, dtype=np.float32)
            return QU
        
    def cl_aa(self):
        """
        Compute the Cl_AA power spectrum for the anisotropic model.
        """
        L = np.arange(self.lmax + 1)
        assert self.Acb is not None, "Acb should be provided for anisotropic model"
        return self.Acb * 2 * np.pi / ( L**2 + L + 1e-30)
    
    def alpha_map(self, idx: int) -> np.ndarray:
        """
        Generate a map of the rotation angle alpha for the anisotropic model.

        Parameters:
        idx (int): Index for the realization of the CMB map.

        Returns:
        np.ndarray: A map of the rotation angle alpha as a NumPy array.

        Notes:
        The method generates a map of the rotation angle alpha for the anisotropic model.
        The map is generated as a random realization of the Cl_AA power spectrum.
        """
        fname = os.path.join(
            self.libdir,
            f"alpha_N{self.nside}_{str(self.Acb).replace('.','p')}_{idx:03d}.fits",
        )
        if os.path.isfile(fname):
            return hp.read_map(fname)
        else:
            cl_aa = self.cl_aa()
            cl_aa[0] = 0
            alm = hp.synalm(cl_aa, lmax=self.lmax,new=True)
            alpha = hp.alm2map(alm, self.nside)
            hp.write_map(fname, alpha, dtype=np.float64)
            return alpha # type: ignore
    
    def get_aniso_cb_lensed_QU(self, idx: int) -> List[np.ndarray]:
        """
        Generate the Q and U Stokes maps after applying cosmic birefringence for the anisotropic model.

        Parameters:
        idx (int): Index for the realization of the CMB map.

        Returns:
        List[np.ndarray]: A list containing the Q and U Stokes parameter maps as NumPy arrays.

        Notes:
        The method applies a rotation to the E and B mode spherical harmonics to simulate the effect of cosmic birefringence.
        If the map for the given `idx` exists in the specified directory, it reads the map from the file.
        Otherwise, it generates the Q and U maps, applies the birefringence, and saves the resulting map to a FITS file.
        """
        fname = os.path.join(
            self.libdir,
            f"cmbQU_N{self.nside}_{str(self.Acb).replace('.','p')}_{idx:03d}.fits",
        )
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])
        else:
            spectra = self.get_lensed_spectra(dl=False)
            T, Q, U = hp.synfast(
                [spectra["tt"], spectra["ee"], spectra["bb"], spectra["te"]],
                nside=self.nside,
                new=True,
            )
            del T
            alpha = self.alpha_map(idx)
            rQ = Q * np.cos(2 * alpha) - U * np.sin(2 * alpha)
            rU = Q * np.sin(2 * alpha) + U * np.cos(2 * alpha)
            del (Q, U)
            hp.write_map(fname, [rQ, rU], dtype=np.float64)
            return [rQ, rU]
        
    