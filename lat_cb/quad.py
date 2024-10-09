import numpy as np
import healpy as hp
import os
from tqdm import tqdm
from lat_cb.signal import LATsky
from lat_cb import mpi
import curvedsky as cs
import pickle as pl
import matplotlib.pyplot as plt


#only for Carlos' EB studies
from lat_cb import cmbEB

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret
class QE:
    path = "/pscratch/sd/c/chervias/Cosmic_birefringence/Forecast_anisotropic_LAT"
    nell_fname = "Nell_fullLAT_5.0_deg_gal_with-srcs-galplane80_spin2.npy"
    mask_fname = "Mask_fullLAT_5.0_deg_gal_with-srcs-galplane80.fits"

    def __init__(self,libdir):
        self.libdir = os.path.join(libdir,'QE')
        if mpi.rank == 0:
            os.makedirs(self.libdir,exist_ok=True)
        mpi.barrier()
        self.nside = 2048
        self.lmax = 4096
        self.Tcmb = 2.726e6
        self.cl_len = (cmbEB.all_cls_th[:self.lmax+1]/self.Tcmb**2).T
        self.beam = hp.gauss_beam(np.deg2rad(2/60),lmax=self.lmax)
        self.fsky = np.mean(self.__get_mask__())

    
    
    def get_noise_cl(self,version=0):
        nell = np.load(os.path.join(self.path,self.nell_fname)).mean(axis=0)
        if version == 0:
            return nell[0],nell[-1]
        elif version == 1:
            ne,nb = nell[0]/self.Tcmb**2,nell[-1]/self.Tcmb**2
            return np.reshape(np.array((cli(ne[:self.lmax+1]),
                          cli(nb[:self.lmax+1]))),(2,1,self.lmax+1))
        else:
            raise ValueError("Invalid version")
    
    def __get_mask__(self,version=0):
        mask = hp.read_map(os.path.join(self.path,self.mask_fname))
        mask[mask > 0] = 1 #binary mask
        if version == 0:
            return mask
        elif version == 1:
            return np.reshape(np.array((mask,mask)),(2,1,hp.nside2npix(self.nside)))
        else:
            raise ValueError("Invalid version")
    
    def __HILC_EB_fname__(self,idx):
        if idx <= 300:
            fname = os.path.join(self.path,"Set1",f"CMB_hilc_LAT_d10s5_Set1_seed{idx:04}.fits")
        else:
            fname = os.path.join(self.path,"Set2",f"CMB_hilc_LAT_d10s5_Set2_seed{idx:04}.fits")

        return fname
    
    def _HILC_EB_(self,idx):
        Emap,Bmap = hp.read_map(self.__HILC_EB_fname__(idx),field=[0,1])
        return hp.map2alm(Emap,lmax=self.lmax),hp.map2alm(Bmap,lmax=self.lmax)

    def QU(self,idx,version=0):
        mask = self.__get_mask__()
        E,B = self._HILC_EB_(idx)
        TQU = hp.alm2map([np.zeros_like(E),E,B],nside=self.nside)*mask
        if version == 0:
            return TQU[1:]
        elif version == 1:
            Q, U = TQU[1:]
            QU = np.reshape(np.array((Q,U)),(2,1,hp.nside2npix(self.nside)))/self.Tcmb
            return QU
        else:
            raise ValueError("Invalid version")
        
    

    def cinv_EB(self,idx,test=False):
        fname = os.path.join(self.libdir,f"cinv_EB_{idx:04d}.pkl")
        if not os.path.isfile(fname):
            QU = self.QU(idx,version=1)
            iterations = [50]
            stat_file = 'stat.txt' 
            if test:
                print(f"Cinv filtering is testing {idx}")
                iterations = [10]
                stat_file = os.path.join(self.libdir,'test_stat.txt')
            
            ninv = self.__get_mask__(version=1)
            Bl = np.reshape(self.beam,(1,self.lmax+1))
            Nl = self.get_noise_cl(version=1)

            E,B = cs.cninv.cnfilter_freq(2,1,self.nside,self.lmax,self.cl_len[1:3,:],
                                        Bl, ninv,QU,chn=1,itns=iterations,filter="",
                                        eps=[1e-5],ro=10,inl=Nl,stat=stat_file)
            if not test:
                pl.dump((E,B),open(fname,'wb'))
        else:
            E,B = pl.load(open(fname,'rb'))
        
        return E,B
    
    def plot_cinv(self,idx):
        """
        plot the cinv filtered Cls for a given idx

        Parameters
        ----------
        idx : int : index of the simulation
        """
        E,B = self.cinv_EB(idx)
        ne,nb = self.get_noise_cl(version=0)
        clb = cs.utils.alm2cl(self.lmax,B)
        cle = cs.utils.alm2cl(self.lmax,E)
        plt.figure(figsize=(4,4))
        plt.loglog(clb,label='B')
        plt.loglog(cle,label='E')
        plt.loglog(1/(self.cl_len[2,:]),label='B theory')
        plt.loglog(1/(self.cl_len[1,:]),label='E theory')
        plt.legend()
        
        

        





    

    