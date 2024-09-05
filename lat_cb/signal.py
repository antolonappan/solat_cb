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
from typing import Dict, Optional, Any, Union, List, Tuple


def inrad(alpha: float) -> float:
    """
    Converts an angle from degrees to radians.

    Parameters:
    alpha (float): The angle in degrees.

    Returns:
    float: The angle in radians.
    """
    return np.deg2rad(alpha)

def cli(cl: np.ndarray) -> np.ndarray:
    """
    Computes the inverse of each element in the input array `cl`.

    Parameters:
    cl (np.ndarray): Input array for which the inverse is calculated.
                     Only positive values will be inverted; zeros and negative values will remain zero.

    Returns:
    np.ndarray: An array where each element is the inverse of the corresponding element in `cl`,
                with zeros or negative values left unchanged.
    """
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1.0 / cl[np.where(cl > 0)]
    return ret

def SO_LAT_Nell(
    sensitivity_mode: int,
    f_sky: float,
    ell_max: int,
) -> Dict[str, np.ndarray]:
    """
    Calculate the noise power spectrum for the SO LAT experiment.

    Parameters:
    sensitivity_mode (int): The sensitivity mode of the experiment.
                            Should be either 1 or 2.
    f_sky (float): The fraction of the sky observed by the experiment.
                     Should be in the range (0, 1].
    ell_max (int): The maximum multipole value for the noise power spectrum.

    Returns:
    Dict[str, np.ndarray]: A dictionary containing the noise power spectrum for each frequency band.
    """

    assert sensitivity_mode == 1 or sensitivity_mode == 2
    assert f_sky > 0.0 and f_sky <= 1.0
    assert ell_max <= 2e4
    NTubes_LF = 1
    NTubes_MF = 4
    NTubes_UHF = 2

    S_LA_27 = np.array([1.0e9, 48.0, 35.0]) * np.sqrt(1.0 / NTubes_LF)
    S_LA_39 = np.array([1.0e9, 24.0, 18.0]) * np.sqrt(1.0 / NTubes_LF)
    S_LA_93 = np.array([1.0e9, 5.4, 3.9]) * np.sqrt(4.0 / NTubes_MF)
    S_LA_145 = np.array([1.0e9, 6.7, 4.2]) * np.sqrt(4.0 / NTubes_MF)
    S_LA_225 = np.array([1.0e9, 15.0, 10.0]) * np.sqrt(2.0 / NTubes_UHF)
    S_LA_280 = np.array([1.0e9, 36.0, 25.0]) * np.sqrt(2.0 / NTubes_UHF)

    f_knee_pol_LA_27 = 700.0
    f_knee_pol_LA_39 = 700.0
    f_knee_pol_LA_93 = 700.0
    f_knee_pol_LA_145 = 700.0
    f_knee_pol_LA_225 = 700.0
    f_knee_pol_LA_280 = 700.0
    alpha_pol = -1.4

    ####################################################################
    ## calculate the survey area and time
    survey_time = 5.0  # years
    t = survey_time * 365.25 * 24.0 * 3600.0  ## convert years to seconds
    t = t * 0.2  ## retention after observing efficiency and cuts
    t = t * 0.85  ## a kludge for the noise non-uniformity of the map edges
    A_SR = 4.0 * np.pi * f_sky  ## sky areas in steradians
    A_deg = A_SR * (180 / np.pi) ** 2  ## sky area in square degrees
    # print("sky area: ", A_deg, "degrees^2")

    ####################################################################
    ## make the ell array for the output noise curves
    ell = np.arange(2, ell_max, 1)

    ####################################################################
    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27 = S_LA_27[sensitivity_mode] / np.sqrt(t)
    W_T_39 = S_LA_39[sensitivity_mode] / np.sqrt(t)
    W_T_93 = S_LA_93[sensitivity_mode] / np.sqrt(t)
    W_T_145 = S_LA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_LA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_LA_280[sensitivity_mode] / np.sqrt(t)

    ####################################################################
    ###   CALCULATE N(ell) for Polarization
    ## calculate the atmospheric contribution for P
    AN_P_27 = (ell / f_knee_pol_LA_27) ** alpha_pol + 1.0
    AN_P_39 = (ell / f_knee_pol_LA_39) ** alpha_pol + 1.0
    AN_P_93 = (ell / f_knee_pol_LA_93) ** alpha_pol + 1.0
    AN_P_145 = (ell / f_knee_pol_LA_145) ** alpha_pol + 1.0
    AN_P_225 = (ell / f_knee_pol_LA_225) ** alpha_pol + 1.0
    AN_P_280 = (ell / f_knee_pol_LA_280) ** alpha_pol + 1.0

    ## calculate N(ell)
    N_ell_P_27 = (W_T_27 * np.sqrt(2)) ** 2.0 * A_SR * AN_P_27
    N_ell_P_39 = (W_T_39 * np.sqrt(2)) ** 2.0 * A_SR * AN_P_39
    N_ell_P_93 = (W_T_93 * np.sqrt(2)) ** 2.0 * A_SR * AN_P_93
    N_ell_P_145 = (W_T_145 * np.sqrt(2)) ** 2.0 * A_SR * AN_P_145
    N_ell_P_225 = (W_T_225 * np.sqrt(2)) ** 2.0 * A_SR * AN_P_225
    N_ell_P_280 = (W_T_280 * np.sqrt(2)) ** 2.0 * A_SR * AN_P_280

    # include cross-correlations due to atmospheric noise
    # use correlation coefficient of r=0.9 within each dichroic pair and 0 otherwise
    r_atm = 0.9
    # different approach than for T -- need to subtract off the white noise part to get the purely atmospheric part
    # see Sec. 2.2 of the SO science goals paper
    N_ell_P_27_atm = (
        (W_T_27 * np.sqrt(2)) ** 2.0 * A_SR * (ell / f_knee_pol_LA_27) ** alpha_pol
    )
    N_ell_P_39_atm = (
        (W_T_39 * np.sqrt(2)) ** 2.0 * A_SR * (ell / f_knee_pol_LA_39) ** alpha_pol
    )
    N_ell_P_93_atm = (
        (W_T_93 * np.sqrt(2)) ** 2.0 * A_SR * (ell / f_knee_pol_LA_93) ** alpha_pol
    )
    N_ell_P_145_atm = (
        (W_T_145 * np.sqrt(2)) ** 2.0 * A_SR * (ell / f_knee_pol_LA_145) ** alpha_pol
    )
    N_ell_P_225_atm = (
        (W_T_225 * np.sqrt(2)) ** 2.0 * A_SR * (ell / f_knee_pol_LA_225) ** alpha_pol
    )
    N_ell_P_280_atm = (
        (W_T_280 * np.sqrt(2)) ** 2.0 * A_SR * (ell / f_knee_pol_LA_280) ** alpha_pol
    )
    N_ell_P_27x39 = r_atm * np.sqrt(N_ell_P_27_atm * N_ell_P_39_atm)
    N_ell_P_93x145 = r_atm * np.sqrt(N_ell_P_93_atm * N_ell_P_145_atm)
    N_ell_P_225x280 = r_atm * np.sqrt(N_ell_P_225_atm * N_ell_P_280_atm)

    ## make a dictionary of noise curves for P
    N_ell_P_LA = {
        "ell": ell,
        "27": N_ell_P_27,
        "39": N_ell_P_39,
        "27x39": N_ell_P_27x39,
        "93": N_ell_P_93,
        "145": N_ell_P_145,
        "93x145": N_ell_P_93x145,
        "225": N_ell_P_225,
        "280": N_ell_P_280,
        "225x280": N_ell_P_225x280,
    }

    return N_ell_P_LA

class CMB:

    def __init__(
        self,
        libdir: str,
        nside: int,
        beta: Optional[float] = None,
        Acb: Optional[float] = None,
        model: str = "iso",
    ):
        """
        Initialize the CMB class for handling Cosmic Microwave Background (CMB) simulations.

        Parameters:
        libdir (str): Directory where the CMB data will be stored.
        nside (int): Resolution parameter for the HEALPix map.
        beta (Optional[float]): Parameter for the isotropic model, should be provided if model is 'iso'.
        Acb (Optional[float]): Parameter for the anisotropic model, should be provided if model is 'aniso'.
        model (str): Model type, either 'iso' for isotropic or 'aniso' for anisotropic. Defaults to 'iso'.

        Attributes:
        libdir (str): Directory where CMB-related data is stored.
        nside (int): HEALPix resolution parameter.
        alpha (Optional[float]): Alpha parameter for isotropic model.
        lmax (int): Maximum multipole moment, computed as 3 * nside - 1.
        powers (Any): Loaded or computed power spectra.
        Acb (Optional[float]): Acb parameter for anisotropic model.
        model (str): The model type ('iso' or 'aniso').

        Raises:
        AssertionError: If required parameters for the chosen model are not provided.
        NotImplementedError: If the 'aniso' model is selected, as it is not implemented yet.
        """

        self.libdir = os.path.join(libdir, "CMB")
        os.makedirs(self.libdir, exist_ok=True)
        self.nside  = nside
        self.beta   = beta
        self.lmax   = 3 * nside - 1
        spectra     = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spectra.pkl")
        if os.path.isfile(spectra):
            self.powers = pl.load(open(spectra, "rb"))
        else:
            self.powers = self.compute_powers()
        self.Acb    = Acb
        assert model in ["iso", "aniso"], "model should be 'iso' or 'aniso'"
        self.model  = model
        if model == "iso":
            assert beta is not None, "beta should be provided for isotropic model"
        if model == "aniso":
            assert Acb is not None, "Acb should be provided for anisotropic model"
        if self.model == "aniso":
            raise NotImplementedError("Anisotropic model is not implemented yet")

    def compute_powers(self) -> Dict[str, Any]:
        """
        compute the CMB power spectra using CAMB.
        """
        ini_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "cb.ini")
        spectra  = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spectra.pkl")
        params   = camb.read_ini(ini_file)
        results  = camb.get_results(params)
        powers   = {}
        powers["cls"] = results.get_cmb_power_spectra(
            params, CMB_unit="muK", raw_cl=True
        )
        powers["dls"] = results.get_cmb_power_spectra(
            params, CMB_unit="muK", raw_cl=False
        )
        if mpi.rank == 0:
            pl.dump(powers, open(spectra, "wb"))
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
        self, beta: float = 0.3, dl: bool = True, dtype: str = "d", new: bool = False
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

    def get_cb_lensed_QU(self, idx: int) -> List[np.ndarray]:
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
            return hp.read_map(fname, field=[0, 1])  # type: ignore
        else:
            spectra = self.get_cb_lensed_spectra(
                beta=self.beta,
                dl=False,
            )
            T, E, B = hp.synalm(
                [spectra["tt"], spectra["ee"], spectra["bb"], spectra["te"], spectra["eb"], spectra["tb"]],
                lmax=self.lmax,
                new=True,
            )
            del T
            QU = hp.alm2map_spin([E, B], self.nside, 2, lmax=self.lmax)
            hp.write_map(fname, QU, dtype=np.float64)
            return QU

class BandpassInt:
    def __init__(
        self,
    ):
        """
        Initializes the BandpassInt class, loading bandpass profiles from a specified file.
        """
        bp_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bp_profile.pkl")
        self.bp = pl.load(open(bp_file, "rb"))

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
        self.libdir = os.path.join(libdir, "Foregrounds")
        os.makedirs(self.libdir, exist_ok=True)
        self.nside = nside
        self.dust_model = dust_model
        self.sync_model = sync_model
        self.bandpass = bandpass
        self.bp_profile = BandpassInt() if bandpass else None

    def dustQU(self, band: Union[str, int]) -> np.ndarray:
        """
        Generates or retrieves the Q and U Stokes parameters for dust emission at a given frequency band.

        Parameters:
        band (Union[str, int]): The frequency band. Can be an integer or a string (e.g., '93GHz').

        Returns:
        np.ndarray: A NumPy array containing the Q and U maps.
        """
        if isinstance(band, (str, np.str_)):
            band = int(band[:-1])
        name = (
            f"dustQU_N{self.nside}_f{band}.fits"
            if not self.bandpass
            else f"dustQU_N{self.nside}_f{band}_bp.fits"
        )
        fname = os.path.join(self.libdir, name)

        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])
        else:
            sky = pysm3.Sky(
                nside=self.nside, preset_strings=[f"d{int(self.dust_model)}"]
            )
            if self.bandpass:
                nu, weights = self.bp_profile.get_profile(band)
                nu = nu * u.GHz
                maps = sky.get_emission(nu, weights)
            else:
                maps = sky.get_emission(band * u.GHz)

            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band * u.GHz))

            if mpi.rank == 0:
                hp.write_map(fname, maps[1:], dtype=np.float64)
            mpi.barrier()

            return maps[1:].value

    def syncQU(self, band: Union[str, int]) -> np.ndarray:
        """
        Generates or retrieves the Q and U Stokes parameters for synchrotron emission at a given frequency band.

        Parameters:
        band (Union[str, int]): The frequency band. Can be an integer or a string (e.g., '93GHz').

        Returns:
        np.ndarray: A NumPy array containing the Q and U maps.
        """
        if isinstance(band, (str, np.str_)):
            band = int(band[:-1])
        name = (
            f"syncQU_N{self.nside}_f{band}.fits"
            if not self.bandpass
            else f"syncQU_N{self.nside}_f{band}_bp.fits"
        )
        fname = os.path.join(self.libdir, name)

        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1])
        else:
            sky = pysm3.Sky(
                nside=self.nside, preset_strings=[f"s{int(self.sync_model)}"]
            )
            if self.bandpass:
                nu, weights = self.bp_profile.get_profile(band)
                nu = nu * u.GHz
                maps = sky.get_emission(nu, weights)
            else:
                maps = sky.get_emission(band * u.GHz)

            maps = maps.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(band * u.GHz))

            if mpi.rank == 0:
                hp.write_map(fname, maps[1:], dtype=np.float64)
            mpi.barrier()

            return maps[1:].value

class Noise:
    nlevp = np.array([71, 36, 8, 10, 22, 54, 71, 36, 8, 10, 22, 54])

    def __init__(self, nside: int, atm_noise: bool = False, nhits: bool = False):
        """
        Initializes the Noise class for generating noise maps with or without atmospheric noise.

        Parameters:
        nside (int): HEALPix resolution parameter.
        atm_noise (bool, optional): If True, includes atmospheric noise. Defaults to False.
        """
        self.nside = nside
        self.lmax = 3 * nside - 1
        self.sensitivity_mode = 2
        self.atm_noise = atm_noise
        if atm_noise:
            self.fsky = 0.6  # Example value for f_sky; adjust as necessary
            self.Nell = SO_LAT_Nell(self.sensitivity_mode, self.fsky, self.lmax)
            print("Noise Model: Atmospheric noise")
        else:
            print("Noise Model: White noise")
        
        self.mask = Mask(nside=self.nside).get_mask(False)
        self.hits_map = Mask(nside=self.nside).get_mask(True)
        self.nhits = nhits
        self.fac = self.get_fac()
    
    def get_fac(self):
        fac = np.sqrt(2)*self.mask
        if self.nhits:
            print("HITS map: enabled")
            grt_zero = self.hits_map > 0
            self.hits_map[grt_zero] /= np.median(self.hits_map[grt_zero])
            with np.errstate(divide='ignore'):
                fac[grt_zero] /= np.sqrt(self.hits_map[grt_zero])
        return fac

    @property
    def rand_alm(self) -> np.ndarray:
        """
        Generates random spherical harmonic coefficients (alm) with a specified power spectrum.

        Returns:
        np.ndarray: A complex array of spherical harmonic coefficients.
        """
        cl = np.repeat(1.0e3, self.lmax + 1)
        return hp.almxfl(hp.synalm(cl, lmax=self.lmax, new=True), 1 / np.sqrt(cl))

    @property
    def cholesky_matrix_elements(
        self,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """
        Computes the Cholesky matrix elements for the noise model.

        Returns:
        Tuple of nine np.ndarray elements representing the Cholesky matrix components.
        """
        L11 = np.zeros(self.lmax, dtype=float)
        L11[2:] = np.sqrt(self.Nell["27"])
        L21 = np.zeros(self.lmax, dtype=float)
        L21[2:] = self.Nell["27x39"] / np.sqrt(self.Nell["27"])
        L22 = np.zeros(self.lmax, dtype=float)
        L22[2:] = np.sqrt(
            (self.Nell["27"] * self.Nell["39"] - self.Nell["27x39"] ** 2)
            / self.Nell["27"]
        )

        L33 = np.zeros(self.lmax, dtype=float)
        L33[2:] = np.sqrt(self.Nell["93"])
        L43 = np.zeros(self.lmax, dtype=float)
        L43[2:] = self.Nell["93x145"] / np.sqrt(self.Nell["93"])
        L44 = np.zeros(self.lmax, dtype=float)
        L44[2:] = np.sqrt(
            (self.Nell["93"] * self.Nell["145"] - self.Nell["93x145"] ** 2)
            / self.Nell["93"]
        )

        L55 = np.zeros(self.lmax, dtype=float)
        L55[2:] = np.sqrt(self.Nell["225"])
        L65 = np.zeros(self.lmax, dtype=float)
        L65[2:] = self.Nell["225x280"] / np.sqrt(self.Nell["225"])
        L66 = np.zeros(self.lmax, dtype=float)
        L66[2:] = np.sqrt(
            (self.Nell["225"] * self.Nell["280"] - self.Nell["225x280"] ** 2)
            / self.Nell["225"]
        )

        return L11, L21, L22, L33, L43, L44, L55, L65, L66

    def atm_noise_maps(self) -> np.ndarray:
        """
        Generates atmospheric noise maps using Cholesky decomposition.

        Returns:
        np.ndarray: An array of atmospheric noise maps for different frequency bands.
        """
        L11, L21, L22, L33, L43, L44, L55, L65, L66 = self.cholesky_matrix_elements

        alm = self.rand_alm
        blm = self.rand_alm
        nlm_27 = hp.almxfl(alm, L11)
        nlm_39 = hp.almxfl(alm, L21) + hp.almxfl(blm, L22)
        n_27 = hp.alm2map(nlm_27, self.nside, pixwin=False)
        n_39 = hp.alm2map(nlm_39, self.nside, pixwin=False)

        clm = self.rand_alm
        dlm = self.rand_alm
        nlm_93 = hp.almxfl(clm, L33)
        nlm_145 = hp.almxfl(clm, L43) + hp.almxfl(dlm, L44)
        n_93 = hp.alm2map(nlm_93, self.nside, pixwin=False)
        n_145 = hp.alm2map(nlm_145, self.nside, pixwin=False)

        elm = self.rand_alm
        flm = self.rand_alm
        nlm_225 = hp.almxfl(elm, L55)
        nlm_280 = hp.almxfl(elm, L65) + hp.almxfl(flm, L66)
        n_225 = hp.alm2map(nlm_225, self.nside, pixwin=False)
        n_280 = hp.alm2map(nlm_280, self.nside, pixwin=False)

        return np.array([n_27, n_39, n_93, n_145, n_225, n_280])

    def noiseQU(self) -> np.ndarray:
        """
        Generates Q and U polarization noise maps based on the noise model.

        Returns:
        np.ndarray: An array of Q and U noise maps.
        """
        if self.atm_noise:
            N = self.noiseQUatm()
        else:
            N = self.noiseQUwhite()
        return N

    def noiseQUwhite(self) -> np.ndarray:
        """
        Generates white noise Q and U polarization maps.

        Returns:
        np.ndarray: An array of white noise Q and U maps.
        """
        depth_p = self.nlevp  # Assuming the first six values are used
        depth_i = depth_p / np.sqrt(2)
        pix_amin2 = (
            4.0 * np.pi / float(hp.nside2npix(self.nside)) * (180.0 * 60.0 / np.pi) ** 2
        )
        sigma_pix_I = np.sqrt(depth_i**2 / pix_amin2)
        sigma_pix_P = np.sqrt(depth_p**2 / pix_amin2)
        npix = hp.nside2npix(self.nside)
        noise = np.random.randn(len(depth_i), 3, npix)
        noise[:, 0, :] *= sigma_pix_I[:, None]
        noise[:, 1, :] *= sigma_pix_P[:, None]
        noise[:, 2, :] *= sigma_pix_P[:, None]
        return noise[:, 1:, :] * self.fac

    def noiseQUatm(self) -> np.ndarray:
        """
        Generates Q and U polarization maps with atmospheric noise.

        Returns:
        np.ndarray: An array of Q and U maps with atmospheric noise.
        """
        q1 = self.atm_noise_maps()
        u1 = self.atm_noise_maps()
        q2 = self.atm_noise_maps()
        u2 = self.atm_noise_maps()

        qu = []
        for i in range(len(q1)):
            qu.append([q1[i], u1[i]])
        for i in range(len(q2)):
            qu.append([q2[i], u2[i]])
        return np.array(qu) * self.fac

class LATsky:
    freqs = np.array(
        [
            "27a",
            "39a",
            "93a",
            "145a",
            "225a",
            "280a",
            "27b",
            "39b",
            "93b",
            "145b",
            "225b",
            "280b",
        ]
    )
    fwhm = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9, 7.4, 5.1, 2.2, 1.4, 1.0, 0.9])
    nlevp = np.array([71, 36, 8, 10, 22, 54, 71, 36, 8, 10, 22, 54])

    def __init__(
        self,
        libdir: str,
        nside: int,
        beta: float,
        dust: int,
        synch: int,
        alpha: Union[float, List[float]],
        atm_noise: bool = False,
        nhits: bool = False,
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
        bandpass (bool, optional): If True, applies bandpass integration. Defaults to False.
        """
        fldname     = "_atm_noise" if atm_noise else ""
        fldname    += "_nhits" if nhits else ""
        self.libdir = os.path.join(libdir, "LAT" + fldname)
        os.makedirs(self.libdir, exist_ok=True)


        configs = {
            self.freqs[i]: {"fwhm": self.fwhm[i], "nlevp": self.nlevp[i]} for i in range(len(self.freqs))
        }
        self.config     = configs
        self.nside      = nside
        self.beta       = beta
        self.cmb        = CMB(libdir, nside, beta)
        self.foreground = Foreground(libdir, nside, dust, synch, bandpass)
        self.dust_model = dust
        self.sync_model = synch
        self.nhits      = nhits
        self.noise      = Noise(nside, atm_noise, nhits)

        if isinstance(alpha, list):
            assert len(alpha) == len(
                self.freqs
            ), "Length of alpha list must match the number of frequency bands."
            for i, a in enumerate(alpha):
                self.config[self.freqs[i]]["alpha"] = a
        else:
            for f in self.freqs:
                self.config[f]["alpha"] = alpha

        self.alpha     = alpha
        self.mask      = Mask(nside).get_mask(False)
        self.atm_noise = atm_noise
        self.bandpass  = bandpass
        if bandpass:
            print("Bandpass is enabled")

    def signalOnlyQU(self, idx: int, band: Union[str, int]) -> np.ndarray:
        """
        Generates the Q and U Stokes parameters for the given frequency band, combining CMB, dust, and synchrotron signals.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        band (Union[str, int]): The frequency band. Can be an integer or a string (e.g., '93GHz').

        Returns:
        np.ndarray: A NumPy array containing the combined Q and U maps.
        """
        if isinstance(band, (str, np.str_)):
            band = int(band[:-1])
        cmbQU  = np.array(self.cmb.get_cb_lensed_QU(idx))
        dustQU = self.foreground.dustQU(band)
        syncQU = self.foreground.syncQU(band)
        return cmbQU + dustQU + syncQU

    def obsQUwAlpha(
        self, idx: int, band: Union[str, int], fwhm: float, alpha: float
    ) -> np.ndarray:
        """
        Generates the observed Q and U Stokes parameters after applying a rotation by alpha and smoothing with a Gaussian beam.

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
        bl     = hp.gauss_beam(np.radians(fwhm / 60), lmax=self.cmb.lmax)
        hp.almxfl(Elm, bl, inplace=True)
        hp.almxfl(Blm, bl, inplace=True)
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
        fwhm  = self.config[band]["fwhm"]
        alpha = self.config[band]["alpha"]
        beta = self.cmb.beta
        return os.path.join(
            self.libdir,
            f"obsQU_N{self.nside}_b{str(beta).replace('.','p')}_a{str(alpha).replace('.','p')}_{band}{'_bp' if self.bandpass else ''}_{fwhm}_{idx:03d}.fits",
        )

    def saveObsQUs(self, idx: int) -> None:
        """
        Saves the observed Q and U Stokes parameter maps for all frequency bands.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        """
        signal = []
        for band in self.freqs:
            fwhm  = self.config[band]["fwhm"]
            alpha = self.config[band]["alpha"]
            signal.append(self.obsQUwAlpha(idx, band, fwhm, alpha))
        noise = self.noise.noiseQU()
        sky   = np.array(signal) + noise
        for i in tqdm(range(len(self.freqs)), desc="Saving Observed QUs", unit="band"):
            band  = self.freqs[i]
            fname = self.obsQUfname(idx, band)
            hp.write_map(fname, sky[i]*self.mask, dtype=np.float64)

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

class Mask:
    def __init__(self, nside: int, libdir: Optional[str] = None) -> None:
        """
        Initializes the Mask class for handling and generating sky masks.

        Parameters:
        nside (int): HEALPix resolution parameter.
        libdir (Optional[str], optional): Directory where the mask may be saved or loaded from. Defaults to None.
        """
        self.nside = nside
        self.libdir = libdir
        self.mask_save = libdir is not None
        self.fsky = np.mean(self.get_mask(nhits=False))

    def get_mask(self, nhits: bool = False) -> np.ndarray:
        """
        Retrieves or generates a sky mask.

        Parameters:
        nhits (bool, optional): If True, returns the hit count map instead of a binary mask. Defaults to False.

        Returns:
        np.ndarray: The mask as a NumPy array. If nhits is False, returns a binary mask; otherwise, returns the hit count map.
        """
        mask = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "mask_N1024.fits"
        )
        ivar = hp.read_map(mask)

        if self.nside != hp.get_nside(ivar):
            ivar = hp.ud_grade(ivar, self.nside)

        if nhits:
            return ivar
        else:
            return (ivar > 0).astype(int)
