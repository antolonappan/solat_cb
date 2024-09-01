# object oriented version of Patricia's code
import numpy as np
import healpy as hp
import pymaster as nmt
import os
from tqdm import tqdm
from lat_cb.signal import  Foreground, LATsky
from lat_cb import mpi

class Spectra:
    def __init__(self,libdir,lat_lib):
        self.lat = lat_lib
        self.nside = self.lat.nside
        self.fg = Foreground(libdir,self.nside,self.lat.dust,self.lat.synch,False)
        fldname = ''
        if self.lat.atm_noise:
            fldname += '_atm'
        self.lmax = 3*self.lat.nside-1
        libdiri = os.path.join(libdir,f'spectra_{self.nside}'+fldname)
        comdir = os.path.join(libdir,f'spectra_{self.nside}'+'_common')
        self.__set_dir__(libdiri,comdir)
        self.binInfo = nmt.NmtBin.from_lmax_linear(self.lmax, 1)
        self.Nell = self.binInfo.get_n_bands()
        self.mask = self.lat.mask
        if not self.lat.nhits:
            self.fsky = np.average(self.mask)
        else:
            raise NotImplementedError("nhits not implemented")
        self.bands = LATsky.freqs
        self.Nbands = len(self.bands)

        self.obs_qu_maps   = None
        self.dust_qu_maps  = None
        self.sync_qu_maps  = None

        self.bandpass = self.lat.bandpass

        self.workspace = nmt.NmtWorkspace()
        self.get_coupling_matrix()

    
    def get_coupling_matrix(self):
        fsky = np.round(self.fsky,2)
        fname = os.path.join(self.wdir, f"coupling_matrix_Nside{self.nside}_fsky_{str(fsky).replace('.','p')}.fits")
        if not os.path.isfile(fname):
            print("Computing coupling Matrix")
            mask_f = nmt.NmtField(self.mask,[self.mask,self.mask],lmax=self.lmax, purify_b=False)
            self.workspace.compute_coupling_matrix(mask_f, mask_f, self.binInfo)
            del mask_f
            self.workspace.write_to(fname)
            print(f"Coupling Matrix saved to {fname}")
        else:
            print(f"Reading coupling matrix from {fname}")
            self.workspace.read_from(fname)
    
    def compute_master(self,f_a, f_b):
        cl_coupled   = nmt.compute_coupled_cell(f_a, f_b)
        cl_decoupled = self.workspace.decouple_cell(cl_coupled)
        return cl_decoupled

    def __set_dir__(self,dir,cdir):
        self.oxo_dir = os.path.join(dir,'obs_x_obs')
        self.dxo_dir = os.path.join(dir,'dust_x_obs')
        self.sxo_dir = os.path.join(dir,'sync_x_obs')
        self.dxd_dir = os.path.join(cdir,'dust_x_dust')
        self.sxs_dir = os.path.join(cdir,'sync_x_sync')
        self.sxd_dir = os.path.join(cdir,'sync_x_dust')
        self.wdir = os.path.join(cdir,'workspaces')
        if mpi.rank==0:
            os.makedirs(self.oxo_dir, exist_ok=True)
            os.makedirs(self.dxo_dir, exist_ok=True)
            os.makedirs(self.sxo_dir, exist_ok=True)
            os.makedirs(self.dxd_dir, exist_ok=True)
            os.makedirs(self.sxs_dir, exist_ok=True)
            os.makedirs(self.sxd_dir, exist_ok=True)
            os.makedirs(self.wdir, exist_ok=True)
        mpi.barrier()

    

    def load_obsQUmaps(self,idx):
        maps = np.zeros((self.Nbands,2,hp.nside2npix(self.nside)),dtype=np.float64)
        for i,band in enumerate(self.bands):
            maps[i] = self.lat.obsQU(idx,band)
        self.obs_qu_maps = maps
    

    def __get_dust_QUmap__(self,band):
        fname = os.path.join(self.fg.libdir, f'dustQU_N{self.nside}_{band}_wbeam.fits')
        if os.path.isfile(fname):
            dust_m = hp.read_map(fname, field=(0,1))
            return dust_m[0], dust_m[1]
        else:
            dust_m = self.fg.dustQU(band)
            E,B = hp.map2alm_spin(dust_m, 2)
            fwhm = LATsky.configs[band]['fwhm']
            bl = hp.gauss_beam(np.radians(fwhm/60), lmax=self.lat.cmb.lmax)
            hp.almxfl(E,bl,inplace=True)
            hp.almxfl(B,bl,inplace=True)
            dust_m = hp.alm2map_spin([E,B], self.nside, 2,self.lat.cmb.lmax)
            hp.write_map(fname, dust_m, dtype=np.float64)
        return dust_m[0], dust_m[1]
    
    def load_dustQUmaps(self):
        maps = np.zeros((self.Nbands,2,hp.nside2npix(self.nside)),dtype=np.float64)
        for i,band in enumerate(self.bands):
            maps[i] = self.__get_dust_QUmap__(band)
        self.dust_qu_maps = maps


    def __get_sync_QUmap__(self,band):
        fname = os.path.join(self.fg.libdir, f'syncQU_N{self.nside}_{band}_wbeam.fits')
        if os.path.isfile(fname):
            sync_m = hp.read_map(fname, field=(0,1))
            return sync_m[0], sync_m[1]
        else:
            sync_m = self.fg.syncQU(band)
            E,B = hp.map2alm_spin([sync_m[0], sync_m[1]], 2)
            fwhm = LATsky.configs[band]['fwhm']
            bl = hp.gauss_beam(np.radians(fwhm/60), lmax=self.lat.cmb.lmax)
            hp.almxfl(E,bl,inplace=True)
            hp.almxfl(B,bl,inplace=True)
            sync_m = hp.alm2map_spin([E,B], self.nside, 2, self.lat.cmb.lmax)
            hp.write_map(fname, sync_m, dtype=np.float64)
            return sync_m[0], sync_m[1]
    
    def load_syncQUmaps(self):
        maps = np.zeros((self.Nbands,2,hp.nside2npix(self.nside)),dtype=np.float64)
        for i,band in enumerate(self.bands):
            maps[i] = self.__get_sync_QUmap__(band)
        self.sync_qu_maps = maps
    
    def __obs_x_obs_helper__(self, ii, idx):
        fname = os.path.join(self.oxo_dir, f"obs_x_obs_{self.bands[ii]}{'_bp' if self.bandpass else ''}_{idx:03d}.npy")
        if os.path.isfile(fname):
            return np.load(fname)
        else:
            cl   = np.zeros((self.Nbands, self.Nbands, 3, self.Nell+2), dtype=np.float64)
            fp_i = nmt.NmtField(self.mask, self.obs_qu_maps[ii], lmax=self.lmax, purify_b=False)
            for jj in range(ii, self.Nbands, 1):
                fp_j  = nmt.NmtField(self.mask, self.obs_qu_maps[jj, :, :], lmax=self.lmax, purify_b=False)

                cl_ij = self.compute_master(fp_i, fp_j)# (EiEj, EiBj, BiEj, BiBj)
                
                cl[ii, jj, 0, 2:] = cl_ij[0, :] # EiEj
                cl[ii, jj, 1, 2:] = cl_ij[3, :] # BiBj
                cl[ii, jj, 2, 2:] = cl_ij[1, :] # EiBj

                if ii!=jj:
                    cl[jj, ii, 0, 2:] = cl_ij[0, :] # EjEi = EiEj
                    cl[jj, ii, 1, 2:] = cl_ij[3, :] # BjBi = BiBj 
                    cl[jj, ii, 2, 2:] = cl_ij[2, :] # EjBi 

                del fp_j
            np.save(fname, cl)
            return cl
        
    def obs_x_obs(self, idx,progress=False):
        cl   = np.zeros((self.Nbands, self.Nbands, 3, self.Nell+2), dtype=np.float64)
        for ii in tqdm(range(self.Nbands),desc='obs x obs spectra',unit='band',disable=not progress):
            cl += self.__obs_x_obs_helper__(ii, idx)
        return cl
    
    def __dust_x_obs_helper__(self, ii, idx):
        fname = os.path.join(self.dxo_dir, f"dust_x_obs_{self.bands[ii]}{'_bp' if self.bandpass else ''}_{idx:03d}.npy")
        if os.path.isfile(fname):
            return np.load(fname)
        else:
            cl   = np.zeros((self.Nbands, self.Nbands, 4, self.Nell+2), dtype=np.float64)
            fp_i = nmt.NmtField(self.mask, self.dust_qu_maps[ii], lmax=self.lmax, purify_b=False)
            for jj in range(0, self.Nbands, 1):
                fp_j  = nmt.NmtField(self.mask, self.obs_qu_maps[jj], lmax=self.lmax, purify_b=False)

                
                cl_ij = self.compute_master(fp_i, fp_j,)# (EiEj, EiBj, BiEj, BiBj)
                
                cl[ii, jj, 0, 2:] = cl_ij[0, :] # EiEj
                cl[ii, jj, 1, 2:] = cl_ij[3, :] # BiBj
                cl[ii, jj, 2, 2:] = cl_ij[1, :] # EiBj
                cl[ii, jj, 3, 2:] = cl_ij[2, :] # BiEj

                del fp_j
            np.save(fname, cl)
            return cl
    
    def dust_x_obs(self, idx,progress=False):
        cl   = np.zeros((self.Nbands, self.Nbands, 4, self.Nell+2), dtype=np.float64)
        for ii in tqdm(range(self.Nbands),desc='dust x obs spectra',unit='band',disable=not progress):
            cl += self.__dust_x_obs_helper__(ii, idx)
        return cl
    
    def __sync_x_obs_helper__(self, ii, idx):
        fname = os.path.join(self.sxo_dir, f"sync_x_obs_{self.bands[ii]}{'_bp' if self.bandpass else ''}_{idx:03d}.npy")
        if os.path.isfile(fname):
            return np.load(fname)
        else:
            cl   = np.zeros((self.Nbands, self.Nbands, 4, self.Nell+2), dtype=np.float64) 
            fp_i = nmt.NmtField(self.mask, self.sync_qu_maps[ii, :, :], lmax=self.lmax, purify_b=False)
            # print(f'sync {bands[ii]}')
            # calculate the full triangle
            for jj in range(0, self.Nbands, 1):
                fp_j  = nmt.NmtField(self.mask, self.obs_qu_maps[jj, :, :], lmax=self.lmax, purify_b=False)
                cl_ij = self.compute_master(fp_i, fp_j)# (EiEj, EiBj, BiEj, BiBj)
                
                cl[ii, jj, 0, 2:] = cl_ij[0, :] # EiEj
                cl[ii, jj, 1, 2:] = cl_ij[3, :] # BiBj
                cl[ii, jj, 2, 2:] = cl_ij[1, :] # EiBj
                cl[ii, jj, 3, 2:] = cl_ij[2, :] # BiEj

                del fp_j

            np.save(fname, cl)
            return cl
    
    def sync_x_obs(self, idx,progress=False):
        cl   = np.zeros((self.Nbands, self.Nbands, 4, self.Nell+2), dtype=np.float64)
        for ii in tqdm(range(self.Nbands),desc='sync x obs spectra',unit='band',disable=not progress):
            cl += self.__sync_x_obs_helper__(ii, idx)
        return cl
    

    def __sync_x_sync_helper__(self, ii):
        fname = os.path.join(self.sxs_dir, f'sync_x_sync_{self.bands[ii]}.npy')
        if os.path.isfile(fname):
            return np.load(fname)
        else:
            cl   = np.zeros((self.Nbands, self.Nbands, 3, self.Nell+2), dtype=np.float64)
            fp_i = nmt.NmtField(self.mask, self.sync_qu_maps[ii, :, :], lmax=self.lmax, purify_b=False)
            for jj in range(ii, self.Nbands, 1):
                fp_j  = nmt.NmtField(self.mask, self.sync_qu_maps[jj, :, :], lmax=self.lmax, purify_b=False)

                
                cl_ij = self.compute_master(fp_i, fp_j)
                
                cl[ii, jj, 0, 2:] = cl_ij[0, :] # EiEj
                cl[ii, jj, 1, 2:] = cl_ij[3, :] # BiBj
                cl[ii, jj, 2, 2:] = cl_ij[1, :] # EiBj

                if ii!=jj:
                    cl[jj, ii, 0, 2:] = cl_ij[0, :] # EjEi = EiEj
                    cl[jj, ii, 1, 2:] = cl_ij[3, :] # BjBi = BiBj 
                    cl[jj, ii, 2, 2:] = cl_ij[2, :] # EjBi 

                del fp_j
            np.save(fname, cl)
            return cl
        
    def sync_x_sync(self,progress=False):
        cl   = np.zeros((self.Nbands, self.Nbands, 3, self.Nell+2), dtype=np.float64)
        for ii in tqdm(range(self.Nbands),desc='sync x sync spectra',unit='band',disable=not progress):
            cl += self.__sync_x_sync_helper__(ii)
        return cl
    
    def __dust_x_dust_helper__(self, ii):
        fname = os.path.join(self.dxd_dir, f'dust_x_dust_{self.bands[ii]}.npy')
        if os.path.isfile(fname):
            return np.load(fname)
        else:
            cl   = np.zeros((self.Nbands, self.Nbands, 3, self.Nell+2), dtype=np.float64) # file format: cl.shape = (band_i, band_j, EE/BB/EB, ell)
            fp_i = nmt.NmtField(self.mask, self.dust_qu_maps[ii, :, :], lmax=self.lmax, purify_b=False)
            # only need to calculate the upper triangle (ii, jj>=ii), the lower triangle is symmetric
            for jj in range(ii,self.Nbands, 1):
                fp_j  = nmt.NmtField(self.mask, self.dust_qu_maps[jj, :, :], lmax=self.lmax, purify_b=False)

                
                cl_ij = self.compute_master(fp_i, fp_j,)# (EiEj, EiBj, BiEj, BiBj)
                
                cl[ii, jj, 0, 2:] = cl_ij[0, :] # EiEj
                cl[ii, jj, 1, 2:] = cl_ij[3, :] # BiBj
                cl[ii, jj, 2, 2:] = cl_ij[1, :] # EiBj

                if ii!=jj:
                    cl[jj, ii, 0, 2:] = cl_ij[0, :] # EjEi = EiEj
                    cl[jj, ii, 1, 2:] = cl_ij[3, :] # BjBi = BiBj 
                    cl[jj, ii, 2, 2:] = cl_ij[2, :] # EjBi 

                del fp_j
            np.save(fname, cl)
            return cl
    
    def dust_x_dust(self,progress=False):
        cl   = np.zeros((self.Nbands, self.Nbands, 3, self.Nell+2), dtype=np.float64)
        for ii in tqdm(range(self.Nbands),desc='dust x dust spectra',unit='band',disable=not progress):
            cl += self.__dust_x_dust_helper__(ii)
        return cl
    
    def __sync_x_dust_helper__(self, ii):
        fname = os.path.join(self.sxd_dir, f'sync_x_dust_{self.bands[ii]}.npy')
        if os.path.isfile(fname):
            return np.load(fname)
        else:
            cl   = np.zeros((self.Nbands, self.Nbands, 4, self.Nell+2), dtype=np.float64) # file format: cl.shape = (band_i, band_j, EE/BB/EB, ell)
            fp_i = nmt.NmtField(self.mask, self.sync_qu_maps[ii, :, :], lmax=self.lmax, purify_b=False)
            # calculate the full triangle
            for jj in range(0, self.Nbands, 1):
                fp_j  = nmt.NmtField(self.mask, self.dust_qu_maps[jj, :, :], lmax=self.lmax, purify_b=False)

                
                cl_ij = self.compute_master(fp_i, fp_j,)# (EiEj, EiBj, BiEj, BiBj)
                
                cl[ii, jj, 0, 2:] = cl_ij[0, :] # EiEj
                cl[ii, jj, 1, 2:] = cl_ij[3, :] # BiBj
                cl[ii, jj, 2, 2:] = cl_ij[1, :] # EiBj
                cl[ii, jj, 3, 2:] = cl_ij[2, :] # BiEj

                del fp_j
            np.save(fname, cl)
            return cl
    
    def sync_x_dust(self,progress=False):
        cl   = np.zeros((self.Nbands, self.Nbands, 4, self.Nell+2), dtype=np.float64)
        for ii in tqdm(range(self.Nbands),desc='sync x dust spectra',unit='band',disable=not progress):
            cl += self.__sync_x_dust_helper__(ii)
        return cl

    def clear_obs_qu_maps(self):
        self.obs_qu_maps = None

    def clear_dust_qu_maps(self):
        self.dust_qu_maps = None

    def clear_sync_qu_maps(self):
        self.sync_qu_maps = None

    def compute(self,idx):
        self.load_obsQUmaps(idx)
        oxo = self.obs_x_obs(idx,progress=True)
        self.load_dustQUmaps()
        dxo = self.dust_x_obs(idx,progress=True)
        dxd = self.dust_x_dust(progress=True)
        self.load_syncQUmaps()
        sxd = self.sync_x_dust(progress=True)
        self.clear_dust_qu_maps()
        sxs = self.sync_x_sync(progress=True)
        sxo = self.sync_x_obs(idx,progress=True)
        self.clear_obs_qu_maps()
        self.clear_sync_qu_maps()
    
    def get_spectra(self,idx):
        oxo = self.obs_x_obs(idx)
        dxo = self.dust_x_obs(idx)
        sxo = self.sync_x_obs(idx)
        dxd = self.dust_x_dust()
        sxs = self.sync_x_sync()
        sxd = self.sync_x_dust()
        return oxo, dxo, dxd, sxd, sxs, sxo

    
