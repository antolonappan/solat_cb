# object oriented version of Patricia's code
import numpy as np
import healpy as hp
import pymaster as nmt
import os
from tqdm import tqdm
from workspace.solat_cb.solat_cb.simulation.signal import LATsky
from workspace.solat_cb.solat_cb.simulation.signal import Foreground
from solat_cb import mpi
from typing import Dict, Optional, Any, Union, List, Tuple


# PDP: eventually we might want to also mask Galactic dust
#TODO PDP: cls are calculated in series not parallel
# each helper should be sent to a different process to accelerate calculation

class Spectra:
    def __init__(self, lat_lib: LATsky, 
                 aposcale: float = 2.0, template_bandpass: bool = False, pureB: bool = False,
                 CO: bool = True, PS: bool = True):
        """
        Initializes the Spectra class for computing and handling power spectra of observed CMB maps.

        Parameters:
        libdir (str): Directory where the spectra will be stored.
        lat_lib (LATsky): An instance of the LATsky class containing LAT-related configurations.
        aposcale (float, optional): Apodisation scale in degrees. Defaults to 2 deg
        template_bandpass (bool, optional): Apply bandpass integration to the foreground template. Defaults to False.
        pureB (bool, optional): Apply B-mode purification. Defaults to False
        CO (bool, optional): Mask the brightest regions of CO emission. Defautls to True.
        PS (bool, optional): Mask the brightest polarised extragalactic point sources. Defaults to True.
        """
        self.lat   = lat_lib
        self.nside = self.lat.nside
        libdir     = self.lat.libdir
        fldname    = "_atm_noise" if self.lat.atm_noise else "_white_noise"
        libdiri    = os.path.join(libdir, f"spectra_{self.nside}_aposcale{str(aposcale).replace('.','p')}{'_pureB' if pureB else ''}" + fldname)
        comdir     = os.path.join(libdir, f"spectra_{self.nside}_aposcale{str(aposcale).replace('.','p')}{'_pureB' if pureB else ''}" + "_common")
        self.__set_dir__(libdiri, comdir)
        
        # PDP: we won't need all these multipoles but I'll leave it like this for now
        self.lmax     = 3 * self.lat.nside - 1
        
        self.temp_bp  = template_bandpass
        self.fg       = Foreground(self.lat.foreground.libdir, self.nside, self.lat.dust_model, self.lat.sync_model, self.temp_bp)
        
        #TODO PDP: We might need some binning, let me test it
        self.binInfo  = nmt.NmtBin.from_lmax_linear(self.lmax, 1)
        self.Nell     = self.binInfo.get_n_bands()
        self.pureB    = pureB
        self.aposcale = aposcale
        self.CO       = CO
        self.PS       = PS
        self.mask     = self.get_apodised_mask()
        self.fsky     = np.mean(self.mask**2)**2/np.mean(self.mask**4)
        
        # PDP: saving the spectra in this order makes the indexing of the mle easier
        self.freqs = LATsky.freqs
        self.Nfreq = len(self.freqs)
        self.bands = []
        for nu in self.freqs:
            for split in range(self.lat.nsplits):
                self.bands.append(f'{nu}-{split+1}')
        self.Nbands = len(self.bands)
        
        self.obs_qu_maps  = None
        self.dust_qu_maps = None
        self.sync_qu_maps = None

        self.workspace = nmt.NmtWorkspace()
        self.get_coupling_matrix()
        
        
    def get_apodised_mask(self) -> np.ndarray:
        fname = os.path.join(
            self.wdir,
            f"mask_N{self.nside}_aposcale{str(self.aposcale).replace('.','p')}{'_CO' if self.CO else ''}{'_PS' if self.PS else ''}.fits",
        )
        if not os.path.isfile(fname):
            bin_mask = np.copy(self.lat.mask)
            if self.CO:
                print("Masking CO")
                co_file = os.path.join(
                            os.path.dirname(os.path.realpath(__file__)), "binary_CO_mask_N1024.fits"
                            )
                co_mask   = hp.read_map(co_file)
                bin_mask *= ( hp.ud_grade(co_mask, self.nside) > 0).astype(int)
                del co_mask
            if self.PS:
                print("Masking PS")
                ps_file = os.path.join(
                            os.path.dirname(os.path.realpath(__file__)), "binary_comb_PS_mask_N1024.fits"
                            )
                ps_mask   = hp.read_map(ps_file)
                bin_mask *= ( hp.ud_grade(ps_mask, self.nside) > 0).astype(int)
                del ps_mask
            print("Apodising mask")
            mask = nmt.mask_apodization(bin_mask, self.aposcale, apotype="C2")
            print(f"Apodised mask saved to {fname}")
            hp.write_map(fname, mask, dtype=float)
            return mask
        else:
            print(f"Reading apodised mask from {fname}")
            return hp.read_map(fname, dtype=float)
        
        
    def get_coupling_matrix(self) -> None:
        """
        Computes or loads the coupling matrix for power spectrum estimation.
        """
        fsky  = np.round(self.fsky, 2)
        fname = os.path.join(
            self.wdir,
            f"coupling_matrix_N{self.nside}_fsky{str(fsky).replace('.','p')}_aposcale{str(self.aposcale).replace('.','p')}{'_CO' if self.CO else ''}{'_PS' if self.PS else ''}{'_pureB' if self.pureB else ''}.fits",
        )
        if not os.path.isfile(fname):
            print("Computing coupling Matrix")
            mask_f = nmt.NmtField(
                self.mask, [self.mask, self.mask], lmax=self.lmax, purify_b=self.pureB
            )
            self.workspace.compute_coupling_matrix(mask_f, mask_f, self.binInfo)
            del mask_f
            self.workspace.write_to(fname)
            print(f"Coupling Matrix saved to {fname}")
        else:
            print(f"Reading coupling matrix from {fname}")
            self.workspace.read_from(fname)

    def compute_master(self, f_a: nmt.NmtField, f_b: nmt.NmtField) -> np.ndarray:
        """
        Computes the decoupled power spectrum using the MASTER algorithm.

        Parameters:
        f_a (nmt.NmtField): First NmtField object.
        f_b (nmt.NmtField): Second NmtField object.

        Returns:
        np.ndarray: Decoupled power spectrum.
        """
        cl_coupled   = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = self.workspace.decouple_cell(cl_coupled)
        return cl_decoupled

    def __set_dir__(self, idir: str, cdir: str) -> None:
        """
        Sets up directories for storing power spectra and workspaces.

        Parameters:
        dir (str): Directory for specific spectra.
        cdir (str): Common directory for spectra and workspaces.
        """
        self.oxo_dir = os.path.join(idir,  "obs_x_obs")
        self.dxo_dir = os.path.join(idir,  "dust_x_obs")
        self.sxo_dir = os.path.join(idir,  "sync_x_obs")
        self.dxd_dir = os.path.join(cdir, "dust_x_dust")
        self.sxs_dir = os.path.join(cdir, "sync_x_sync")
        self.sxd_dir = os.path.join(cdir, "sync_x_dust")
        self.wdir    = os.path.join(cdir, "workspaces")
        if mpi.rank == 0:
            os.makedirs(self.oxo_dir, exist_ok=True)
            os.makedirs(self.dxo_dir, exist_ok=True)
            os.makedirs(self.sxo_dir, exist_ok=True)
            os.makedirs(self.dxd_dir, exist_ok=True)
            os.makedirs(self.sxs_dir, exist_ok=True)
            os.makedirs(self.sxd_dir, exist_ok=True)
            os.makedirs(self.wdir,    exist_ok=True)
        mpi.barrier()

    def load_obsQUmaps(self, idx: int) -> None:
        """
        Loads observed Q and U Stokes parameter maps for all frequency bands.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        """
        maps = np.zeros((self.Nbands, 2, hp.nside2npix(self.nside)), dtype=np.float64)
        for i, band in enumerate(self.bands):
            maps[i] = self.lat.obsQU(idx, band)
        self.obs_qu_maps = maps

    def __get_fg_QUmap__(self, nu: str, fg: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves or generates the Q and U Stokes parameter maps for dust emission for a specific frequency band.

        Parameters:
        band (str): The frequency identifier.
        fg (str): Foreground type, either 'dust' or 'sync'
        Returns:
        Tuple[np.ndarray, np.ndarray]: Q and U maps for dust emission.
        """
        if fg not in ['dust', 'sync']:
            raise ValueError('Unknown foreground')
        fname = os.path.join(self.fg.libdir, f"{fg}QU_N{self.nside}_{nu}_template{'_bp' if self.temp_bp else ''}.fits")
        if os.path.isfile(fname):
            m = hp.read_map(fname, field=(0, 1))
            return m[0], m[1]
        else:
            if fg=='dust':
                m = self.fg.dustQU(nu)
            elif fg=='sync':
                m = self.fg.syncQU(nu)
            E, B   = hp.map2alm_spin(m, 2, lmax=self.lmax)
            fwhm   = self.lat.fwhm[self.freqs==nu][0]
            bl     = hp.gauss_beam(np.radians(fwhm / 60), pol=True, lmax=self.lmax)
            pwf    = np.array(hp.pixwin(self.nside, pol=True, lmax=self.lmax))
            hp.almxfl(E, bl[:,1]*pwf[1,:], inplace=True)
            hp.almxfl(B, bl[:,2]*pwf[1,:], inplace=True)
            m      = hp.alm2map_spin([E, B], self.nside, 2, self.lmax)*self.lat.mask
            hp.write_map(fname, m, dtype=np.float64)
            return m[0], m[1]

    def load_dustQUmaps(self) -> None:
        """
        Loads dust Q and U Stokes parameter maps for all frequency bands.
        """
        maps = np.zeros((self.Nfreq, 2, hp.nside2npix(self.nside)), dtype=np.float64)
        for i, nu in enumerate(self.freqs):
            maps[i] = self.__get_fg_QUmap__(nu, 'dust')
        self.dust_qu_maps = maps

    def load_syncQUmaps(self) -> None:
        """
        Loads synchrotron Q and U Stokes parameter maps for all frequency bands.
        """
        maps = np.zeros((self.Nfreq, 2, hp.nside2npix(self.nside)), dtype=np.float64)
        for i, nu in enumerate(self.freqs):
            maps[i] = self.__get_fg_QUmap__(nu, 'sync')
        self.sync_qu_maps = maps

    def __obs_x_obs_helper__(self, ii: int, idx: int) -> np.ndarray:
        """
        Helper function:
        Computes or loads the observed x observed power spectra for a specific frequency band.

        Parameters:
        ii (int): Index for the current frequency band.
        idx (int): Index for the realization of the CMB map.

        Returns:
        np.ndarray: Power spectra for the observed x observed fields.
        """
        fname = os.path.join(
            self.oxo_dir,
            f"obs_x_obs_{self.bands[ii]}{'_obsBP' if self.lat.bandpass else ''}_{idx:03d}.npy",
        )
        if os.path.isfile(fname):
            return np.load(fname)
        else:
            cl = np.zeros(
                (self.Nbands, self.Nbands, 3, self.Nell + 2), dtype=np.float64
            )
            fp_i = nmt.NmtField(
                self.mask, self.obs_qu_maps[ii], lmax=self.lmax, purify_b=self.pureB,
                masked_on_input=False
            )
            for jj in range(ii, self.Nbands, 1):
                fp_j = nmt.NmtField(
                    self.mask, self.obs_qu_maps[jj], lmax=self.lmax, purify_b=self.pureB,
                    masked_on_input=False
                )

                cl_ij = self.compute_master(fp_i, fp_j)  # (EiEj, EiBj, BiEj, BiBj)

                cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
                cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
                cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj

                if ii != jj:
                    cl[jj, ii, 0, 2:] = cl_ij[0, :]  # EjEi = EiEj
                    cl[jj, ii, 1, 2:] = cl_ij[3, :]  # BjBi = BiBj
                    cl[jj, ii, 2, 2:] = cl_ij[2, :]  # EjBi

                del fp_j
            np.save(fname, cl)
            return cl

    def obs_x_obs(self, idx: int, progress: bool = False) -> np.ndarray:
        """
        Computes or loads the observed x observed power spectra for all frequency bands.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
        np.ndarray: Combined power spectra for the observed x observed fields across all bands.
        """
        cl = np.zeros((self.Nbands, self.Nbands, 3, self.Nell + 2), dtype=np.float64)
        for ii in tqdm(
            range(self.Nbands),
            desc="obs x obs spectra",
            unit="band",
            disable=not progress,
        ):
            cl += self.__obs_x_obs_helper__(ii, idx)
        return cl

    def __fg_x_obs_helper__(self, ii: int, idx: int, fg: str) -> np.ndarray:
        """
        Helper function:
        Computes or loads the dust x observed power spectra for a specific frequency band.

        Parameters:
        ii (int): Index for the current frequency band.
        idx (int): Index for the realization of the CMB map.
        fg (str): Type of foregrounds, either 'dust' or 'sync'
        Returns:
        np.ndarray: Power spectra for the dust x observed fields.
        """
        if fg not in ['dust', 'sync']:
            raise ValueError('Unknown foreground')
            
        if fg=='dust':
            base_dir = self.dxo_dir
        elif fg=='sync':
            base_dir = self.sxo_dir
        fname = os.path.join(base_dir,
            f"{fg}_x_obs_{self.freqs[ii]}{'_obsBP' if self.lat.bandpass else ''}{'_tempBP' if self.temp_bp else ''}_{idx:03d}.npy",
        )
        
        if os.path.isfile(fname): 
            return np.load(fname)
        else:
            cl = np.zeros((self.Nfreq, self.Nbands, 4, self.Nell + 2), dtype=np.float64)
            if fg=='dust':
                fp_i = nmt.NmtField(
                    self.mask, self.dust_qu_maps[ii], lmax=self.lmax, purify_b=self.pureB,
                    masked_on_input=False
                )
            elif fg=='sync':
                fp_i = nmt.NmtField(
                    self.mask, self.sync_qu_maps[ii], lmax=self.lmax, purify_b=self.pureB,
                    masked_on_input=False
                )
            for jj in range(0, self.Nbands, 1):
                fp_j = nmt.NmtField(
                    self.mask, self.obs_qu_maps[jj], lmax=self.lmax, purify_b=self.pureB,
                    masked_on_input=False
                )

                cl_ij = self.compute_master(fp_i,fp_j)  # (EiEj, EiBj, BiEj, BiBj)

                cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
                cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
                cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj
                cl[ii, jj, 3, 2:] = cl_ij[2, :]  # BiEj

                del fp_j
            np.save(fname, cl)
            return cl

    def dust_x_obs(self, idx: int, progress: bool = False) -> np.ndarray:
        """
        Computes or loads the dust x observed power spectra for all frequency bands.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
        np.ndarray: Combined power spectra for the dust x observed fields across all bands.
        """
        cl = np.zeros((self.Nfreq, self.Nbands, 4, self.Nell + 2), dtype=np.float64)
        for ii in tqdm(
            range(self.Nfreq),
            desc="dust x obs spectra",
            unit="band",
            disable=not progress,
        ):
            cl += self.__fg_x_obs_helper__(ii, idx, 'dust')
        return cl

    def sync_x_obs(self, idx: int, progress: bool = False) -> np.ndarray:
        """
        Computes or loads the synchrotron x observed power spectra for all frequency bands.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
        np.ndarray: Combined power spectra for the synchrotron x observed fields across all bands.
        """
        cl = np.zeros((self.Nfreq, self.Nbands, 4, self.Nell + 2), dtype=np.float64)
        for ii in tqdm(
            range(self.Nfreq),
            desc="sync x obs spectra",
            unit="band",
            disable=not progress,
        ):
            cl += self.__fg_x_obs_helper__(ii, idx, 'sync')
        return cl

    def __fg_x_fg_helper__(self, ii: int, fg: str) -> np.ndarray:
        """
        Helper function:
        Computes or loads the synchrotron x synchrotron power spectra for a specific frequency band.

        Parameters:
        ii (int): Index for the current frequency band.
        fg (str): Type of foregrounds, either 'dust' or 'sync'
        Returns:
        np.ndarray: Power spectra for the synchrotron x synchrotron fields.
        """
        if fg not in ['dust', 'sync']:
            raise ValueError('Unknown foreground')
            
        if fg=='dust':
            base_dir = self.dxd_dir
        elif fg=='sync':
            base_dir = self.sxs_dir
        fname = os.path.join(base_dir,
            f"{fg}_x_{fg}_{self.freqs[ii]}{'_tempBP' if self.temp_bp else ''}.npy",
        )
        
        if os.path.isfile(fname):
            return np.load(fname)
        else:
            cl = np.zeros(
                (self.Nfreq, self.Nfreq, 3, self.Nell + 2), dtype=np.float64
            )
            if fg=='dust':
                fp_i = nmt.NmtField(
                    self.mask, self.dust_qu_maps[ii], lmax=self.lmax, purify_b=self.pureB,
                    masked_on_input=False
                )
            elif fg=='sync':
                fp_i = nmt.NmtField(
                    self.mask, self.sync_qu_maps[ii], lmax=self.lmax, purify_b=self.pureB,
                    masked_on_input=False
                )
                
            for jj in range(ii, self.Nfreq, 1):
                if fg=='dust':
                    fp_j = nmt.NmtField(
                        self.mask, self.dust_qu_maps[jj], lmax=self.lmax, purify_b=self.pureB,
                        masked_on_input=False
                    )
                elif fg=='sync':
                    fp_j = nmt.NmtField(
                        self.mask, self.sync_qu_maps[jj], lmax=self.lmax, purify_b=self.pureB,
                        masked_on_input=False
                    )

                cl_ij = self.compute_master(fp_i, fp_j)

                cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
                cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
                cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj

                if ii != jj:
                    cl[jj, ii, 0, 2:] = cl_ij[0, :]  # EjEi = EiEj
                    cl[jj, ii, 1, 2:] = cl_ij[3, :]  # BjBi = BiBj
                    cl[jj, ii, 2, 2:] = cl_ij[2, :]  # EjBi

                del fp_j
            np.save(fname, cl)
            return cl

    def sync_x_sync(self, progress: bool = False) -> np.ndarray:
        """
        Computes or loads the synchrotron x synchrotron power spectra for all frequency bands.

        Parameters:
        progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
        np.ndarray: Combined power spectra for the synchrotron x synchrotron fields across all bands.
        """
        cl = np.zeros((self.Nfreq, self.Nfreq, 3, self.Nell + 2), dtype=np.float64)
        for ii in tqdm(
            range(self.Nfreq),
            desc="sync x sync spectra",
            unit="band",
            disable=not progress,
        ):
            cl += self.__fg_x_fg_helper__(ii, 'sync')
        return cl

    def dust_x_dust(self, progress: bool = False) -> np.ndarray:
        """
        Computes or loads the dust x dust power spectra for all frequency bands.

        Parameters:
        progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
        np.ndarray: Combined power spectra for the dust x dust fields across all bands.
        """
        cl = np.zeros((self.Nfreq, self.Nfreq, 3, self.Nell + 2), dtype=np.float64)
        for ii in tqdm(
            range(self.Nfreq),
            desc="dust x dust spectra",
            unit="band",
            disable=not progress,
        ):
            cl += self.__fg_x_fg_helper__(ii, 'dust')
        return cl

    def __sync_x_dust_helper__(self, ii: int) -> np.ndarray:
        """
        Helper function:
        Computes or loads the synchrotron x dust power spectra for a specific frequency band.

        Parameters:
        ii (int): Index for the current frequency band.

        Returns:
        np.ndarray: Power spectra for the synchrotron x dust fields.
        """
        fname = os.path.join(self.sxd_dir, f"sync_x_dust_{self.freqs[ii]}{'_tempBP' if self.temp_bp else ''}.npy")
        if os.path.isfile(fname):
            return np.load(fname)
        else:
            cl = np.zeros(
                (self.Nfreq, self.Nfreq, 4, self.Nell + 2), dtype=np.float64
            )
            fp_i = nmt.NmtField(
                self.mask, self.sync_qu_maps[ii], lmax=self.lmax, purify_b=self.pureB,
                masked_on_input=False
            )
            for jj in range(0, self.Nfreq, 1):
                fp_j = nmt.NmtField(
                    self.mask, self.dust_qu_maps[jj], lmax=self.lmax, purify_b=self.pureB,
                    masked_on_input=False
                )

                cl_ij = self.compute_master(fp_i,fp_j)  # (EiEj, EiBj, BiEj, BiBj)

                cl[ii, jj, 0, 2:] = cl_ij[0, :]  # EiEj
                cl[ii, jj, 1, 2:] = cl_ij[3, :]  # BiBj
                cl[ii, jj, 2, 2:] = cl_ij[1, :]  # EiBj
                cl[ii, jj, 3, 2:] = cl_ij[2, :]  # BiEj

                del fp_j
            np.save(fname, cl)
            return cl 

    def sync_x_dust(self, progress: bool = False) -> np.ndarray:
        """
        Computes or loads the synchrotron x dust power spectra for all frequency bands.

        Parameters:
        progress (bool, optional): If True, displays a progress bar. Defaults to False.

        Returns:
        np.ndarray: Combined power spectra for the synchrotron x dust fields across all bands.
        """
        cl = np.zeros((self.Nfreq, self.Nfreq, 4, self.Nell + 2), dtype=np.float64)
        for ii in tqdm(
            range(self.Nfreq),
            desc="sync x dust spectra",
            unit="band",
            disable=not progress,
        ):
            cl += self.__sync_x_dust_helper__(ii)
        return cl

    def clear_obs_qu_maps(self) -> None:
        """Clears the loaded observed Q and U maps to free up memory."""
        self.obs_qu_maps = None

    def clear_dust_qu_maps(self) -> None:
        """Clears the loaded dust Q and U maps to free up memory."""
        self.dust_qu_maps = None

    def clear_sync_qu_maps(self) -> None:
        """Clears the loaded synchrotron Q and U maps to free up memory."""
        self.sync_qu_maps = None

    def compute(self, idx: int, sync: bool = False) -> None:
        """
        Computes and stores all relevant spectra for a given realization index.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        sync (bool, optional): If True, calculate also synchrotron power spectra. Defaults to False.
        """
        self.load_dustQUmaps()
        dxd = self.dust_x_dust(progress=True)
        self.load_obsQUmaps(idx)
        oxo = self.obs_x_obs(idx, progress=True)
        dxo = self.dust_x_obs(idx, progress=True)
        if sync:
            self.load_syncQUmaps()
            sxd = self.sync_x_dust(progress=True)
            self.clear_dust_qu_maps()
            sxs = self.sync_x_sync(progress=True)
            sxo = self.sync_x_obs(idx, progress=True)
            self.clear_obs_qu_maps()
            self.clear_sync_qu_maps()
            del (oxo, dxo, dxd, sxd, sxs, sxo)
        else:
            self.clear_dust_qu_maps()
            self.clear_obs_qu_maps()
            del (oxo, dxo, dxd)

    def get_spectra(self, idx: int, 
                    sync: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieves all relevant spectra for a given realization index.

        Parameters:
        idx (int): Index for the realization of the CMB map.
        sync (bool, optional): If True, calculate also synchrotron power spectra. Defaults to False.
        Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Tuple containing the power spectra (oxo, dxo, dxd, sxd, sxs, sxo).
        """
        oxo = self.obs_x_obs(idx)
        dxo = self.dust_x_obs(idx)
        dxd = self.dust_x_dust()
        if sync:
            sxo = self.sync_x_obs(idx)
            sxs = self.sync_x_sync()
            sxd = self.sync_x_dust()
            return {'oxo':oxo, 'dxd':dxd, 'sxs':sxs, 'dxo':dxo, 'sxo':sxo, 'sxd':sxd}
        else:
            return {'oxo':oxo, 'dxd':dxd, 'dxo':dxo}
