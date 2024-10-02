import numpy as np
import healpy as hp
import os
from tqdm import tqdm
from solat_cb import mpi
from typing import Union, List, Optional

from solat_cb.simulation import CMB, Foreground, Mask, Noise
from solat_cb.utils import Logger, inrad


class SkySimulation:
    def __init__(
        self,
        libdir: str,
        nside: int,
        cb_method: str,
        dust: int,
        synch: int,
        alpha: Union[float, List[float]],
        freqs: np.ndarray,
        fwhm: np.ndarray,
        tube: np.ndarray,
        beta: Optional[float],
        Acb: Optional[float],
        atm_noise: bool = False,
        nsplits: int = 2,
        bandpass: bool = False,
        verbose: bool = True,
        fldname_suffix: str = "",
    ):
        """
        Initializes the SkySimulation class for generating and handling sky simulations.

        Parameters:
        libdir (str): Directory where the sky maps will be stored.
        nside (int): HEALPix resolution parameter.
        beta (float): Rotation angle for cosmic birefringence in degrees.
        dust (int): Model number for the dust emission.
        synch (int): Model number for the synchrotron emission.
        alpha (Union[float, List[float]]): Polarisation angle(s) for frequency bands. If a list, should match the number of frequency bands.
        atm_noise (bool, optional): If True, includes atmospheric noise. Defaults to False.
        nsplits (int, optional): Number of data splits to consider. Defaults to 2.
        bandpass (bool, optional): If True, applies bandpass integration. Defaults to False.
        verbose (bool, optional): If True, enables verbose output. Defaults to True.
        freqs (np.ndarray, optional): Array of frequency bands.
        fwhm (np.ndarray, optional): Array of full-width half-maximum for the Gaussian beam.
        tube (np.ndarray, optional): Array of tube identifiers.
        fldname_suffix (str, optional): Suffix to append to the folder name. Defaults to "".
        """
        self.logger = Logger(self.__class__.__name__, verbose)
        self.verbose = verbose

        fldname = "_atm_noise" if atm_noise else "_white_noise"
        fldname += "_bandpass" if bandpass else ""
        fldname += f"_{nsplits}splits" + fldname_suffix
        self.basedir = libdir
        self.libdir = os.path.join(libdir, self.__class__.__name__[:3] + fldname)
        os.makedirs(self.libdir + '/obs', exist_ok=True)

        self.nside = nside
        self.Acb = Acb
        self.cb_method = cb_method
        self.beta = beta
        self.cmb = CMB(libdir, nside, beta, Acb, cb_method,self.verbose)
        self.foreground = Foreground(libdir, nside, dust, synch, bandpass, verbose=False)
        self.dust_model = dust
        self.sync_model = synch
        self.nsplits = nsplits
        self.freqs = freqs
        self.fwhm = fwhm
        self.tube = tube
        self.mask, self.fsky = self.__set_mask_fsky__(libdir)
        self.noise = Noise(nside, self.fsky, self.__class__.__name__[:3], atm_noise, nsplits, verbose=self.verbose)
        self.config = {}
        for split in range(nsplits):
            for band in range(len(self.freqs)):
                self.config[f'{self.freqs[band]}-{split+1}'] = {"fwhm": self.fwhm[band], "opt. tube": self.tube[band]}

        if isinstance(alpha, (list, np.ndarray)):
            assert self.freqs is not None and len(alpha) == len(
                self.freqs
            ), "Length of alpha list must match the number of frequency bands."
            for band, a in enumerate(alpha):
                for split in range(self.nsplits):
                    self.config[f'{self.freqs[band]}-{split+1}']["alpha"] = a
        else:
            if self.freqs is not None:
                for split in range(self.nsplits):
                    for band in range(len(self.freqs)):
                        self.config[f'{self.freqs[band]}-{split+1}']["alpha"] = alpha

        self.alpha = alpha
        self.atm_noise = atm_noise
        self.bandpass = bandpass

    def __set_mask_fsky__(self, libdir):
        maskobj = Mask(libdir, self.nside, self.__class__.__name__[:3], verbose=self.verbose)
        return maskobj.mask, maskobj.fsky

    def signalOnlyQU(self, idx: int, band: str) -> np.ndarray:
        band = band[:band.index('-')]
        cmbQU = np.array(self.cmb.get_cb_lensed_QU(idx))
        dustQU = self.foreground.dustQU(band)
        syncQU = self.foreground.syncQU(band)
        return cmbQU + dustQU + syncQU

    def obsQUwAlpha(
        self, idx: int, band: str, fwhm: float, alpha: float
    ) -> np.ndarray:
        signal = self.signalOnlyQU(idx, band)
        E, B = hp.map2alm_spin(signal, 2, lmax=self.cmb.lmax)
        Elm = (E * np.cos(inrad(2 * alpha))) - (B * np.sin(inrad(2 * alpha)))
        Blm = (E * np.sin(inrad(2 * alpha))) + (B * np.cos(inrad(2 * alpha)))
        del (E, B)
        bl = hp.gauss_beam(inrad(fwhm / 60), lmax=self.cmb.lmax, pol=True)
        pwf = np.array(hp.pixwin(self.nside, pol=True,))
        hp.almxfl(Elm, bl[:, 1] * pwf[1, :], inplace=True)
        hp.almxfl(Blm, bl[:, 2] * pwf[1, :], inplace=True)
        return hp.alm2map_spin([Elm, Blm], self.nside, 2, lmax=self.cmb.lmax)

    def obsQUfname(self, idx: int, band: str) -> str:
        alpha = self.config[band]["alpha"]
        beta = self.cmb.beta
        return os.path.join(
            self.libdir,
            f"obs/obsQU_N{self.nside}_b{str(beta).replace('.','p')}_a{str(alpha).replace('.','p')}_{band}{'_bp' if self.bandpass else ''}_{idx:03d}.fits",
        )

    def saveObsQUs(self, idx: int) -> None:
        bands = list(self.config.keys())
        signal = []
        for band in bands:
            fwhm = self.config[band]["fwhm"]
            alpha = self.config[band]["alpha"]
            signal.append(self.obsQUwAlpha(idx, band, fwhm, alpha))
        noise = self.noise.noiseQU()
        sky = np.array(signal) + noise
        for i in tqdm(range(len(bands)), desc="Saving Observed QUs", unit="band"):
            fname = self.obsQUfname(idx, bands[i])
            hp.write_map(fname, sky[i] * self.mask, dtype=np.float64, overwrite=True) # type: ignore

    def obsQU(self, idx: int, band: str) -> np.ndarray:
        fname = self.obsQUfname(idx, band)
        if os.path.isfile(fname):
            return hp.read_map(fname, field=[0, 1]) # type: ignore
        else:
            self.saveObsQUs(idx)
            return hp.read_map(fname, field=[0, 1]) # type: ignore

class LATsky(SkySimulation):
    freqs = np.array(["27", "39", "93", "145", "225", "280"])
    fwhm = np.array([7.4, 5.1, 2.2, 1.4, 1.0, 0.9])  # arcmin
    tube = np.array(["LF", "LF", "MF", "MF", "HF", "HF"])  # tube each frequency occupies

    def __init__(
        self,
        libdir: str,
        nside: int,
        cb_method: str,
        dust: int,
        synch: int,
        alpha: Union[float, List[float]],
        beta: Optional[float] = None,
        Acb: Optional[float] = None,
        atm_noise: bool = False,
        nsplits: int = 2,
        bandpass: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            libdir=libdir,
            nside=nside,
            cb_method=cb_method,
            dust=dust,
            synch=synch,
            alpha=alpha,
            freqs=LATsky.freqs,
            fwhm=LATsky.fwhm,
            tube=LATsky.tube,
            beta=beta,
            Acb=Acb,
            atm_noise=atm_noise,
            nsplits=nsplits,
            bandpass=bandpass,
            verbose=verbose,
            fldname_suffix="",
        )


class SATsky(SkySimulation):
    freqs = np.array(["30", "40", "100", "150", "220", "280"])
    fwhm = np.array([30.0, 20.0, 10.0, 7.0, 5.0, 3.5])  # example values
    tube = np.array(["S1", "S1", "S2", "S2", "S3", "S3"])  # example tube identifiers

    def __init__(
        self,
        libdir: str,
        nside: int,
        cb_method: str,
        dust: int,
        synch: int,
        alpha: Union[float, List[float]],
        beta: Optional[float] = None,
        Acb: Optional[float] = None,
        atm_noise: bool = False,
        nsplits: int = 2,
        bandpass: bool = False,
        verbose: bool = True,
    ):
        super().__init__(
            libdir=libdir,
            nside=nside,
            cb_method=cb_method,
            dust=dust,
            synch=synch,
            alpha=alpha,
            freqs=SATsky.freqs,
            fwhm=SATsky.fwhm,
            tube=SATsky.tube,
            beta=beta,
            Acb=Acb,
            atm_noise=atm_noise,
            nsplits=nsplits,
            bandpass=bandpass,
            verbose=verbose,
            fldname_suffix="",
        )