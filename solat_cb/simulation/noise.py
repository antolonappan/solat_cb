import numpy as np
import healpy as hp 
from typing import Dict, Optional, Any, Union, List, Tuple

from solat_cb import mpi
from solat_cb.utils import Logger

def SO_LAT_Nell_v3_0_0(
    sensitivity_mode: int,
    f_sky: float,
    ell_max: int,
    atm_noise: bool
) -> Dict[str, np.ndarray]:
    """
    Calculate the noise power spectrum for the SO LAT experiment.
    v3.0.0 model adapted from https://github.com/simonsobs/so_noise_models/blob/master/so_models_v3/SO_Noise_Calculator_Public_v3_0_0.py
    Parameters:
    sensitivity_mode (int): The sensitivity mode of the experiment.
                            Should be either 1 or 2.
    f_sky (float): The fraction of the sky observed by the experiment.
                     Should be in the range (0, 1].
    atm_noise (bool): Return white + 1/f noise if True or white noise only if alse
    ell_max (int): The maximum multipole value for the noise power spectrum.

    Returns:
    Dict[str, np.ndarray]: A dictionary containing the noise power spectrum for each frequency band.
    """

    assert sensitivity_mode == 1 or sensitivity_mode == 2
    assert f_sky > 0.0 and f_sky <= 1.0
    assert ell_max <= 2e4
    NTubes_LF  = 1
    NTubes_MF  = 4
    NTubes_UHF = 2

    S_LA_27  = np.array([1.0e9, 48.0, 35.0]) * np.sqrt(1.0 / NTubes_LF)
    S_LA_39  = np.array([1.0e9, 24.0, 18.0]) * np.sqrt(1.0 / NTubes_LF)
    S_LA_93  = np.array([1.0e9,  5.4,  3.9]) * np.sqrt(4.0 / NTubes_MF)
    S_LA_145 = np.array([1.0e9,  6.7,  4.2]) * np.sqrt(4.0 / NTubes_MF)
    S_LA_225 = np.array([1.0e9, 15.0, 10.0]) * np.sqrt(2.0 / NTubes_UHF)
    S_LA_280 = np.array([1.0e9, 36.0, 25.0]) * np.sqrt(2.0 / NTubes_UHF)

    f_knee_pol_LA_27  = 700.0
    f_knee_pol_LA_39  = 700.0
    f_knee_pol_LA_93  = 700.0
    f_knee_pol_LA_145 = 700.0
    f_knee_pol_LA_225 = 700.0
    f_knee_pol_LA_280 = 700.0
    alpha_pol = -1.4

    ## calculate the survey area and time
    survey_time = 5.0  # years
    t = survey_time * 365.25 * 24.0 * 3600.0  ## convert years to seconds
    t = t * 0.2  ## retention after observing efficiency and cuts
    # PDP: I think we should remove this when providing a hitmap separately
    t = t * 0.85  ## a kludge for the noise non-uniformity of the map edges
    A_SR = 4.0 * np.pi * f_sky  ## sky areas in steradians

    ## make the ell array for the output noise curves
    ell = np.arange(2, ell_max, 1)

    ###   CALCULATE N(ell) for Temperature
    ## calculate the experimental weight
    W_T_27  = S_LA_27[sensitivity_mode] / np.sqrt(t)
    W_T_39  = S_LA_39[sensitivity_mode] / np.sqrt(t)
    W_T_93  = S_LA_93[sensitivity_mode] / np.sqrt(t)
    W_T_145 = S_LA_145[sensitivity_mode] / np.sqrt(t)
    W_T_225 = S_LA_225[sensitivity_mode] / np.sqrt(t)
    W_T_280 = S_LA_280[sensitivity_mode] / np.sqrt(t)

    if not atm_noise:
        ###   CALCULATE N(ell) for Polarization
        NW_P_27   = (W_T_27  * np.sqrt(2))**2.* A_SR 
        NW_P_39   = (W_T_39  * np.sqrt(2))**2.* A_SR 
        NW_P_93   = (W_T_93  * np.sqrt(2))**2.* A_SR 
        NW_P_145  = (W_T_145 * np.sqrt(2))**2.* A_SR 
        NW_P_225  = (W_T_225 * np.sqrt(2))**2.* A_SR 
        NW_P_280  = (W_T_280 * np.sqrt(2))**2.* A_SR 
        
        N_ell_P = {
            "ell": ell,
            "27": np.repeat(NW_P_27, len(ell)),
            "39": np.repeat(NW_P_39, len(ell)),
            "93": np.repeat(NW_P_93, len(ell)),
            "145": np.repeat(NW_P_145, len(ell)),
            "225": np.repeat(NW_P_225, len(ell)),
            "280": np.repeat(NW_P_280, len(ell)),
        }
    else:
        ###   CALCULATE N(ell) for Polarization
        ## calculate the atmospheric contribution for P
        AN_P_27  = (ell / f_knee_pol_LA_27) ** alpha_pol + 1.0
        AN_P_39  = (ell / f_knee_pol_LA_39) ** alpha_pol + 1.0
        AN_P_93  = (ell / f_knee_pol_LA_93) ** alpha_pol + 1.0
        AN_P_145 = (ell / f_knee_pol_LA_145) ** alpha_pol + 1.0
        AN_P_225 = (ell / f_knee_pol_LA_225) ** alpha_pol + 1.0
        AN_P_280 = (ell / f_knee_pol_LA_280) ** alpha_pol + 1.0
    
        ## calculate N(ell)
        N_ell_P_27  = ( W_T_27 * np.sqrt(2)) ** 2.0 * A_SR * AN_P_27
        N_ell_P_39  = ( W_T_39 * np.sqrt(2)) ** 2.0 * A_SR * AN_P_39
        N_ell_P_93  = ( W_T_93 * np.sqrt(2)) ** 2.0 * A_SR * AN_P_93
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
        N_ell_P_27x39   = r_atm * np.sqrt( N_ell_P_27_atm * N_ell_P_39_atm)
        N_ell_P_93x145  = r_atm * np.sqrt( N_ell_P_93_atm * N_ell_P_145_atm)
        N_ell_P_225x280 = r_atm * np.sqrt(N_ell_P_225_atm * N_ell_P_280_atm)
    
        ## make a dictionary of noise curves for P
        N_ell_P = {
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

    return N_ell_P



class Noise:

    def __init__(self, 
                 nside: int, 
                 fsky: float, 
                 atm_noise: bool = False, 
                 nsplits: int = 2,
                 verbose: bool = True,
                 ) -> None:
        """
        Initializes the Noise class for generating noise maps with or without atmospheric noise.

        Parameters:
        nside (int): HEALPix resolution parameter.
        atm_noise (bool, optional): If True, includes atmospheric noise. Defaults to False.
        nhits (bool, optional): If True, includes hit count map. Defaults to False.
        nsplits (int, optional): Number of data splits to consider. Defaults to 2.
        """
        self.nside            = nside
        self.lmax             = 3 * nside - 1
        self.sensitivity_mode = 2
        self.atm_noise        = atm_noise
        self.nsplits          = nsplits
        self.Nell             = SO_LAT_Nell_v3_0_0(self.sensitivity_mode, fsky, self.lmax, self.atm_noise)
        self.logger           = Logger(self.__class__.__name__, verbose)
        if atm_noise:
            self.logger.log("Noise Model: White + 1/f noise v3.0.0")
        else:
            self.logger.log("Noise Model: White noise v3.0.0")

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
        L11     = np.zeros(self.lmax, dtype=float)
        L11[2:] = np.sqrt(self.Nell["27"])
        L21     = np.zeros(self.lmax, dtype=float)
        L21[2:] = self.Nell["27x39"] / np.sqrt(self.Nell["27"])
        L22     = np.zeros(self.lmax, dtype=float)
        L22[2:] = np.sqrt(
            (self.Nell["27"] * self.Nell["39"] - self.Nell["27x39"] ** 2)
            / self.Nell["27"]
        )

        L33     = np.zeros(self.lmax, dtype=float)
        L33[2:] = np.sqrt(self.Nell["93"])
        L43     = np.zeros(self.lmax, dtype=float)
        L43[2:] = self.Nell["93x145"] / np.sqrt(self.Nell["93"])
        L44     = np.zeros(self.lmax, dtype=float)
        L44[2:] = np.sqrt(
            (self.Nell["93"] * self.Nell["145"] - self.Nell["93x145"] ** 2)
            / self.Nell["93"]
        )

        L55     = np.zeros(self.lmax, dtype=float)
        L55[2:] = np.sqrt(self.Nell["225"])
        L65     = np.zeros(self.lmax, dtype=float)
        L65[2:] = self.Nell["225x280"] / np.sqrt(self.Nell["225"])
        L66     = np.zeros(self.lmax, dtype=float)
        L66[2:] = np.sqrt(
            (self.Nell["225"] * self.Nell["280"] - self.Nell["225x280"] ** 2)
            / self.Nell["225"]
        )

        return L11, L21, L22, L33, L43, L44, L55, L65, L66


    def __white_noise__(self, band):
        n = hp.synfast(np.concatenate((np.zeros(2), self.Nell[band])), self.nside, lmax=self.lmax, pixwin=False)
        return n
        
    
    def white_noise_maps(self) -> np.ndarray: 
        return np.array([
                    self.__white_noise__('27'),
                    self.__white_noise__('39'),
                    self.__white_noise__('93'),
                    self.__white_noise__('145'),
                    self.__white_noise__('225'),
                    self.__white_noise__('280')])


    def atm_noise_maps(self) -> np.ndarray:
        """
        Generates atmospheric noise maps using Cholesky decomposition.

        Returns:
        np.ndarray: An array of atmospheric noise maps for different frequency bands.
        """
        L11, L21, L22, L33, L43, L44, L55, L65, L66 = self.cholesky_matrix_elements

        alm    = self.rand_alm
        blm    = self.rand_alm
        nlm_27 = hp.almxfl(alm, L11)
        nlm_39 = hp.almxfl(alm, L21) + hp.almxfl(blm, L22)
        n_27   = hp.alm2map(nlm_27, self.nside, pixwin=False)
        n_39   = hp.alm2map(nlm_39, self.nside, pixwin=False)

        clm     = self.rand_alm
        dlm     = self.rand_alm
        nlm_93  = hp.almxfl(clm, L33)
        nlm_145 = hp.almxfl(clm, L43) + hp.almxfl(dlm, L44)
        n_93    = hp.alm2map(nlm_93, self.nside, pixwin=False)
        n_145   = hp.alm2map(nlm_145, self.nside, pixwin=False)

        elm     = self.rand_alm
        flm     = self.rand_alm
        nlm_225 = hp.almxfl(elm, L55)
        nlm_280 = hp.almxfl(elm, L65) + hp.almxfl(flm, L66)
        n_225   = hp.alm2map(nlm_225, self.nside, pixwin=False)
        n_280   = hp.alm2map(nlm_280, self.nside, pixwin=False)

        n = np.array([n_27, n_39, n_93, n_145, n_225, n_280])
        return n

    def noiseQU(self) -> np.ndarray:
        """
        Generates Q and U polarization noise maps based on the noise model.

        Returns:
        np.ndarray: An array of Q and U noise maps.
        """
        N = []
        for split in range(self.nsplits):
            if self.atm_noise:
                q = self.atm_noise_maps()
                u = self.atm_noise_maps()
            else:
                q = self.white_noise_maps()
                u = self.white_noise_maps()            
              
            for i in range(len(q)):
                N.append([q[i], u[i]])

        return np.array(N)*np.sqrt(self.nsplits)