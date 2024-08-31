# object oriented version of Patricia's code
import numpy as np
import healpy as hp
from lat_cb.spectra import Spectra
from lat_cb.signal import CMB
from lat_cb import mpi

rad2arcmin = 180*60/np.pi


def bin_from_edges(start, end):
    nls  = np.amax(end)
    lmax = nls-1
    ells, bpws, weights = [], [], []
    for ib, (li, le) in enumerate(zip(start, end)):
        nlb      = int(le - li)
        ells    += list(range(li, le))
        bpws    += [ib] * nlb
        weights += [1./nlb] * nlb
    ells = np.array(ells); bpws = np.array(bpws); weights = np.array(weights)

    nell       = len(ells)
    n_bands    = bpws[-1]+1
    nell_array = np.zeros(n_bands, dtype=int)
    for ii in range(0, nell, 1):
        if ells[ii]<=lmax and bpws[ii]>=0:
            nell_array[bpws[ii]]+=1   
            
    ell_list, w_list = [], []
    for ii in range(0, n_bands, 1):
        ell_list.append(np.zeros(nell_array[ii], dtype=int))
        w_list.append(np.zeros(nell_array[ii], dtype=np.float64))
    
    nell_array *= 0
    for ii in range(0, nell, 1):
        if ells[ii]<=lmax and bpws[ii]>=0: 
            ell_list[bpws[ii]][nell_array[bpws[ii]]] = ells[ii]
            w_list[bpws[ii]][nell_array[bpws[ii]]]   = weights[ii]
            nell_array[bpws[ii]]+=1

    for ii in range(0, n_bands, 1):
        norm=0
        for jj in range(0, nell_array[ii], 1):
            norm += w_list[ii][jj]
        if norm<=0:
            print(f"Weights in band {ii} are wrong\n")
        for jj in range(0, nell_array[ii], 1):
            w_list[ii][jj]/=norm

    return (n_bands, nell_array, ell_list, w_list)


def bin_configuration(info):
    #transformar el output de my_binning_from_edges para el caso de binneado uniforme eficiente de my_bin_cls_fast
    (n_bands, nell_array, ell_list, w_list) = info
    ib_array          = np.arange(0, n_bands,       1, dtype=int)
    il_array          = np.arange(0, nell_array[0], 1, dtype=int)
    (ib_grid,il_grid) = np.meshgrid(ib_array, il_array, indexing='ij')
    return (ib_grid, il_grid, np.array(w_list), np.array(ell_list))


def bin_spec_matrix(spec, info):
    (ib_grid, il_grid, w_array, ell_array) = info
    return np.sum(w_array[ib_grid,il_grid]*spec[:,:,ell_array[ib_grid,il_grid]], axis=3)


def bin_cov_matrix(cov, info):
    (ib_grid, il_grid, w_array, ell_array) = info
    return np.sum(w_array[ib_grid,il_grid]**2*cov[:,:,:,ell_array[ib_grid,il_grid]], axis=4)


class MLE:

    def __init__(self,libdir,spec_lib,binwidth=20,bmin=20,bmax=1000):
        self.spec = spec_lib
        self.cmb = CMB(libdir,self.spec.lat.nside,self.spec.lat.alpha)
        self.cmb_cls = self.cmb.get_lensed_spectra(dl=False,dtype='a').T
        self.fsky = np.mean(self.spec.mask)
        self.niter_max =100
        self.nside = self.spec.lat.nside



        self.nlb   = binwidth
        self.bmin  = bmin
        self.bmax  = bmax

        assert bmax <= self.spec.lmax, "bmax must be less than lmax in Spectra object"

        # define binning
        lower_edge = np.arange(self.bmin, self.bmax-self.nlb ,self.nlb)
        upper_edge = np.arange(self.bmin+self.nlb, self.bmax ,self.nlb)
        bin_def    = bin_from_edges(lower_edge, upper_edge)
        self.bin_conf   = bin_configuration(bin_def)
        self.Nbins      = bin_def[0]

        self.inst = {'27a' : {'telescope': 'LFT', 'nu':  '27', 'fwhm': 7.4, 'idx': 0}, 
                     '39a' : {'telescope': 'LFT', 'nu':  '39', 'fwhm': 5.1, 'idx': 1}, 
                     '93a' : {'telescope': 'LFT', 'nu':  '93', 'fwhm': 2.2, 'idx': 2}, 
                     '145a': {'telescope': 'LFT', 'nu': '145', 'fwhm': 1.4, 'idx': 3}, 
                     '225a': {'telescope': 'LFT', 'nu': '225', 'fwhm': 1.0, 'idx': 4}, 
                     '280a': {'telescope': 'LFT', 'nu': '280', 'fwhm': 0.9, 'idx': 5},
                     '27b' : {'telescope': 'LFT', 'nu':  '27', 'fwhm': 7.4, 'idx': 6}, 
                     '39b' : {'telescope': 'LFT', 'nu':  '39', 'fwhm': 5.1, 'idx': 7}, 
                     '93b': {'telescope': 'LFT', 'nu':  '93', 'fwhm': 2.2, 'idx': 8}, 
                     '145b': {'telescope': 'LFT', 'nu': '145', 'fwhm': 1.4, 'idx': 9}, 
                     '225b': {'telescope': 'LFT', 'nu': '225', 'fwhm': 1.0, 'idx': 10}, 
                     '280b': {'telescope': 'LFT', 'nu': '280', 'fwhm': 0.9, 'idx': 11}}

        self.Nbands = self.spec.Nbands
        self.bands = self.spec.bands
        self.dt = np.float64
        self.ExtParam = 4 # As, Ad, Asd, beta + N alpha_i
        self.Nvar  = self.Nbands + self.ExtParam


        self.MNi  = np.zeros((self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1)), dtype=np.uint8)
        self.MNj  = np.zeros((self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1)), dtype=np.uint8)
        self.MNh  = np.zeros((self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1)), dtype=np.uint8)
        self.MNk  = np.zeros((self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1)), dtype=np.uint8)

        std_i = np.zeros((self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), self.bmax+1), dtype=self.dt)
        std_j = np.zeros((self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), self.bmax+1), dtype=self.dt)
        std_h = np.zeros((self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), self.bmax+1), dtype=self.dt)
        std_k = np.zeros((self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), self.bmax+1), dtype=self.dt)


        IJidx = [] 
        for ii in range(0, self.Nbands, 1):
            for jj in range(0, self.Nbands, 1):
                if jj!=ii: # exclude auto-spectra condition
                    IJidx.append((ii,jj))
        self.IJidx = np.array(IJidx, dtype=np.uint8) # data type valid for <=70 bands, optimizing memory use

        MNidx = [] 
        for mm in range(0, self.Nbands*(self.Nbands-1), 1):
            for nn in range(0, self.Nbands*(self.Nbands-1), 1):
                    MNidx.append((mm,nn))
        self.MNidx = np.array(MNidx, dtype=np.uint16) # data type valid for <=70 bands, optimizing memory use

        for MN_pair in self.MNidx:
            ii, jj, hh, kk, mm, nn=self.get_index(MN_pair)
            self.MNi[mm, nn] = ii; self.MNj[mm, nn] = jj
            self.MNh[mm, nn] = hh; self.MNk[mm, nn] = kk
            
            std_i[mm, nn, :] = np.repeat(self.inst[self.bands[ii]]['fwhm'], self.bmax+1)
            std_j[mm, nn, :] = np.repeat(self.inst[self.bands[jj]]['fwhm'], self.bmax+1)
            std_h[mm, nn, :] = np.repeat(self.inst[self.bands[hh]]['fwhm'], self.bmax+1)
            std_k[mm, nn, :] = np.repeat(self.inst[self.bands[kk]]['fwhm'], self.bmax+1)

        self.std_i = np.deg2rad(std_i/60)/(2*np.sqrt(2*np.log(2)))
        self.std_j = np.deg2rad(std_j/60)/(2*np.sqrt(2*np.log(2)))
        self.std_h = np.deg2rad(std_h/60)/(2*np.sqrt(2*np.log(2)))
        self.std_k = np.deg2rad(std_k/60)/(2*np.sqrt(2*np.log(2)))
    
    def get_index(self,mn_pair):
        mm, nn = mn_pair
        ii, jj = self.IJidx[mm]
        hh, kk = self.IJidx[nn]
        return ii, jj, hh, kk, mm, nn

    def convolveCls_gaussBeams_pwf(self,in_cl, fwhm_1, fwhm_2, lmax,):
        ell     = np.arange(0, lmax+1, 1)
        sg_1     = np.deg2rad(fwhm_1/60)/(2*np.sqrt(2*np.log(2)))
        beam_1   = np.exp(-ell*(ell+1)*sg_1**2/2)
        sg_2     = np.deg2rad(fwhm_2/60)/(2*np.sqrt(2*np.log(2)))
        beam_2   = np.exp(-ell*(ell+1)*sg_2**2/2)
        (_, pwf) = hp.pixwin(self.nside, pol=True, lmax=lmax)
        #pixel window function is squared because both are polarization fields
        return in_cl[:lmax+1] * beam_1 * beam_2 * pwf**2


    def build_cov(self,inTo, inTcmb, inTs, inTd, inTSD, inTDS,
                inTs_o, inTd_o, inTSD_o, inTDS_o,
                inTs_d, inTSD_DS, inTs_SD, inTs_DS, inTd_SD, inTd_DS,
                params): 
        # This step shouldn't be necessary but at times I've had problems with overwritten variables during iteration
        To    = np.copy(inTo)   ; Tcmb = np.copy(inTcmb)
        Td    = np.copy(inTd)   ; Ts   = np.copy(inTs)    ; TSD    = np.copy(inTSD)   ; TDS   = np.copy(inTDS)
        Td_o  = np.copy(inTd_o) ; Ts_o = np.copy(inTs_o)  ; TSD_o  = np.copy(inTSD_o) ; TDS_o = np.copy(inTDS_o)
        Ts_d  = np.copy(inTs_d) ; Ts_SD = np.copy(inTs_SD); Ts_DS  = np.copy(inTs_DS)
        Td_SD = np.copy(inTd_SD); Td_DS = np.copy(inTd_DS); TSD_DS = np.copy(inTSD_DS)
        
        # variables ordered like As, Ad, Asd, beta, alpha_i
        As   = params[0]
        Ad   = params[1]
        Asd  = params[2]
        beta = params[3]
        ai   = params[self.MNi+self.ExtParam]; aj = params[self.MNj+self.ExtParam]
        ah   = params[self.MNh+self.ExtParam]; ak = params[self.MNk+self.ExtParam]
        
        # trigonometric factors rotating the spectra
        cicj = np.cos(2*ai)*np.cos(2*aj); sisj = np.sin(2*ai)*np.sin(2*aj)
        c4ij = np.cos(4*ai)+np.cos(4*aj)
    
        chck = np.cos(2*ah)*np.cos(2*ak); shsk = np.sin(2*ah)*np.sin(2*ak)
        c4hk = np.cos(4*ah)+np.cos(4*ak)
        
        Aij = np.sin(4*aj)/c4ij; Ahk = np.sin(4*ak)/c4hk
        Bij = np.sin(4*ai)/c4ij; Bhk = np.sin(4*ah)/c4hk
        Dij = 2*cicj/c4ij      ; Dhk = 2*chck/c4hk
        Eij = 2*sisj/c4ij      ; Ehk = 2*shsk/c4hk
        Cij = np.sin(4*beta)/(2*np.cos(2*ai+2*aj))
        Chk = np.sin(4*beta)/(2*np.cos(2*ah+2*ak))
        
        # covariance elements
        # observed * observed; remove all EB except the one in T0
        Cov  =  To[0,:,:,:] + Ahk*Aij*To[1,:,:,:] + Bhk*Bij*To[2,:,:,:] 
        # cmb * cmb + cmb * observed
        Cov += - 2*Cij*Chk*( Tcmb[0,:,:,:] + Tcmb[1,:,:,:] )
        # synch * synch; remove EB from T1, T2, T3
        Cov += + Dij*Dhk*As**2*Ts[0,:,:,:] + Eij*Ehk*As**2*Ts[1,:,:,:] + Dij*Ehk*As**2*Ts[2,:,:,:] + Eij*Dhk*As**2*Ts[3,:,:,:]
        # dust * dust; remove EB from T1, T2, T3
        Cov += + Dij*Dhk*Ad**2*Td[0,:,:,:] + Eij*Ehk*Ad**2*Td[1,:,:,:] + Dij*Ehk*Ad**2*Td[2,:,:,:] + Eij*Dhk*Ad**2*Td[3,:,:,:]
        # synch-dust * synch-dust; remove EB from T1, T2, T3
        Cov += + Dij*Dhk*Asd**2*TSD[0,:,:,:] + Eij*Ehk*Asd**2*TSD[1,:,:,:] + Dij*Ehk*Asd**2*TSD[2,:,:,:] + Eij*Dhk*Asd**2*TSD[3,:,:,:]
        # dust-synch * dust-synch; remove EB from T1, T2, T3
        Cov += + Dij*Dhk*Asd**2*TDS[0,:,:,:] + Eij*Ehk*Asd**2*TDS[1,:,:,:] + Dij*Ehk*Asd**2*TDS[2,:,:,:] + Eij*Dhk*Asd**2*TDS[3,:,:,:]
        # synch * dust; remove EB from T2, T3, T4, T5, T6, T7
        Cov += + Dij*Dhk*As*Ad*( Ts_d[0,:,:,:] + Ts_d[1,:,:,:] ) + Eij*Ehk*As*Ad*( Ts_d[2,:,:,:] + Ts_d[3,:,:,:] ) 
        Cov += + Dij*Ehk*As*Ad*( Ts_d[4,:,:,:] + Ts_d[7,:,:,:] ) + Dhk*Eij*As*Ad*( Ts_d[5,:,:,:] + Ts_d[6,:,:,:] )
        # synch-dust * dust-synch; remove EB from T2, T3, T4, T5, T6, T7
        Cov += + Dij*Dhk*Asd**2*( TSD_DS[0,:,:,:] + TSD_DS[1,:,:,:] ) + Eij*Ehk*Asd**2*( TSD_DS[2,:,:,:] + TSD_DS[3,:,:,:] )
        Cov += + Dij*Ehk*Asd**2*( TSD_DS[4,:,:,:] + TSD_DS[7,:,:,:] ) + Dhk*Eij*Asd**2*( TSD_DS[5,:,:,:] + TSD_DS[6,:,:,:] )
        # synch * synch-dust; remove EB from T2, T3, T4, T5, T6, T7
        Cov += + Dij*Dhk*As*Asd*( Ts_SD[0,:,:,:] + Ts_SD[1,:,:,:] ) + Eij*Ehk*As*Asd*( Ts_SD[6,:,:,:] + Ts_SD[7,:,:,:] )
        Cov += + Dij*Ehk*As*Asd*( Ts_SD[2,:,:,:] + Ts_SD[5,:,:,:] ) + Dhk*Eij*As*Asd*( Ts_SD[3,:,:,:] + Ts_SD[4,:,:,:] )
        # synch * dust-synch; remove EB from T2, T3, T4, T5, T6, T7
        Cov += + Dij*Dhk*As*Asd*( Ts_DS[0,:,:,:] + Ts_DS[1,:,:,:] ) + Eij*Ehk*As*Asd*( Ts_DS[6,:,:,:] + Ts_DS[7,:,:,:] )
        Cov += + Dij*Ehk*As*Asd*( Ts_DS[2,:,:,:] + Ts_DS[5,:,:,:] ) + Dhk*Eij*As*Asd*( Ts_DS[3,:,:,:] + Ts_DS[4,:,:,:] )
        # dust * synch-dust; remove EB from T2, T3, T4, T5, T6, T7
        Cov += + Dij*Dhk*Ad*Asd*( Td_SD[0,:,:,:] + Td_SD[1,:,:,:] ) + Eij*Ehk*Ad*Asd*( Td_SD[6,:,:,:] + Td_SD[7,:,:,:] )
        Cov += + Dij*Ehk*Ad*Asd*( Td_SD[2,:,:,:] + Td_SD[5,:,:,:] ) + Dhk*Eij*Ad*Asd*( Td_SD[3,:,:,:] + Td_SD[4,:,:,:] )
        # dust * dust-synch; remove EB from T2, T3, T4, T5, T6, T7
        Cov += + Dij*Dhk*Ad*Asd*( Td_DS[0,:,:,:] + Td_DS[1,:,:,:] ) + Eij*Ehk*Ad*Asd*( Td_DS[6,:,:,:] + Td_DS[7,:,:,:] )
        Cov += + Dij*Ehk*Ad*Asd*( Td_DS[2,:,:,:] + Td_DS[5,:,:,:] ) + Dhk*Eij*Ad*Asd*( Td_DS[3,:,:,:] + Td_DS[4,:,:,:] )
        # synch * observed; remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        Cov += - Dij*As*Ts_o[0,:,:,:] - Dhk*As*Ts_o[1,:,:,:]  - Eij*As*Ts_o[2,:,:,:] - Ehk*As*Ts_o[3,:,:,:]
        # dust * observed; remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        Cov += - Dij*Ad*Td_o[0,:,:,:] - Dhk*Ad*Td_o[1,:,:,:] - Eij*Ad*Td_o[2,:,:,:] - Ehk*Ad*Td_o[3,:,:,:]
        # synch-dust * observed; remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        Cov += - Dij*Asd*TSD_o[0,:,:,:] - Dhk*Asd*TSD_o[1,:,:,:] - Eij*Asd*TSD_o[2,:,:,:] - Ehk*Asd*TSD_o[3,:,:,:]
        # dust-synch * observed; remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        Cov += - Dij*Asd*TDS_o[0,:,:,:] - Dhk*Asd*TDS_o[1,:,:,:] - Eij*Asd*TDS_o[2,:,:,:] - Ehk*Asd*TDS_o[3,:,:,:]

        return Cov


    def cov_elements_auto(self,EiEjo, BiBjo, EiBjo, BiEjo,
                        EiEjs, BiBjs, EiBjs, BiEjs,
                        EiEjd, BiBjd, EiBjd, BiEjd,
                        Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd,
                        si, sj, sh, sk,
                        cmb, binInfo, lmax):  
            
        ell      = np.arange(0, lmax+1, 1)
        (_, pwf) = hp.pixwin(self.nside, pol=True, lmax=lmax)
        #####################
        # cmb 
        Tcmb = np.zeros((2, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        #prepare beams       
        beam = np.exp(-ell*(ell+1)*si**2/2)*np.exp(-ell*(ell+1)*sj**2/2)*np.exp(-ell*(ell+1)*sh**2/2)*np.exp(-ell*(ell+1)*sk**2/2)*pwf**4
        del si, sj, sh, sk
        Tcmb[0,:,:,:] = (beam*cmb[1,:self.bmax+1]**2)/(2*ell+1)
        Tcmb[1,:,:,:] = (beam*cmb[2,:self.bmax+1]**2)/(2*ell+1)
        Tcmb = np.moveaxis(bin_cov_matrix(Tcmb, binInfo), 3, 1)
        del cmb # free memory
        ############################ remove all except T0(1), T5(6), T6(7)
        # observed * observed
        To = np.zeros((3, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        #(1) observed
        To[0,:,:,:] = (EiEjo[self.MNi,self.MNh,:]*BiBjo[self.MNj,self.MNk,:] + EiBjo[self.MNi,self.MNk,:]*BiEjo[self.MNj,self.MNh,:])/(2*ell+1)
        #(6) observed
        To[1,:,:,:] = (EiEjo[self.MNi,self.MNh,:]*EiEjo[self.MNj,self.MNk,:] + EiEjo[self.MNi,self.MNk,:]*EiEjo[self.MNj,self.MNh,:])/(2*ell+1)
        #(7) observed
        To[2,:,:,:] = (BiBjo[self.MNi,self.MNh,:]*BiBjo[self.MNj,self.MNk,:] + BiBjo[self.MNi,self.MNk,:]*BiBjo[self.MNj,self.MNh,:])/(2*ell+1)
        To = np.moveaxis(bin_cov_matrix(To, binInfo), 3, 1)
        del EiEjo, BiBjo, EiBjo, BiEjo # free memory
        ##################### remove EB from T1, T2, T3
        # synch * synch 
        Ts = np.zeros((4, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        #(1s)
        Ts[0,:,:,:] = (EiEjs[self.MNi,self.MNh,:]*BiBjs[self.MNj,self.MNk,:] + EiBjs[self.MNi,self.MNk,:]*BiEjs[self.MNj,self.MNh,:])/(2*ell+1)
        #(2s)
        Ts[1,:,:,:] = BiBjs[self.MNi,self.MNh,:]*EiEjs[self.MNj,self.MNk,:]/(2*ell+1) 
        #(3s)
        Ts[2,:,:,:] = EiEjs[self.MNi,self.MNk,:]*BiBjs[self.MNj,self.MNh,:]/(2*ell+1)
        #(4s)
        Ts[3,:,:,:] = BiBjs[self.MNi,self.MNk,:]*EiEjs[self.MNj,self.MNh,:]/(2*ell+1)
        Ts = np.moveaxis(bin_cov_matrix(Ts, binInfo), 3, 1)
        ##################### remove EB from T1, T2, T3
        # dust * dust
        Td = np.zeros((4, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1d)
        Td[0,:,:,:] = (EiEjd[self.MNi,self.MNh,:]*BiBjd[self.MNj,self.MNk,:] + EiBjd[self.MNi,self.MNk,:]*BiEjd[self.MNj,self.MNh,:])/(2*ell+1)
        # (2d)
        Td[1,:,:,:] = BiBjd[self.MNi,self.MNh,:]*EiEjd[self.MNj,self.MNk,:]/(2*ell+1) 
        # (3d)
        Td[2,:,:,:] = EiEjd[self.MNi,self.MNk,:]*BiBjd[self.MNj,self.MNh,:]/(2*ell+1)
        # (4d)
        Td[3,:,:,:] = BiBjd[self.MNi,self.MNk,:]*EiEjd[self.MNj,self.MNh,:]/(2*ell+1)
        Td = np.moveaxis(bin_cov_matrix(Td, binInfo), 3, 1)
        ##################### remove EB from T1, T2, T3
        # synch-dust * synch-dust 
        TSD = np.zeros((4, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1SD)
        TSD[0,:,:,:] = (EiEjs[self.MNi,self.MNh,:]*BiBjd[self.MNj,self.MNk,:] + Eis_Bjd[self.MNi,self.MNk,:]*Eis_Bjd[self.MNh,self.MNj,:])/(2*ell+1)
        # (2SD)
        TSD[1,:,:,:] = BiBjs[self.MNi,self.MNh,:]*EiEjd[self.MNj,self.MNk,:]/(2*ell+1) 
        # (3SD)
        TSD[2,:,:,:] = Eis_Ejd[self.MNi,self.MNk,:]*Bis_Bjd[self.MNh,self.MNj,:]/(2*ell+1)
        # (4SD)
        TSD[3,:,:,:] = Bis_Bjd[self.MNi,self.MNk,:]*Eis_Ejd[self.MNh,self.MNj,:]/(2*ell+1)
        TSD = np.moveaxis(bin_cov_matrix(TSD, binInfo), 3, 1)
        ##################### remove EB from T1, T2, T3
        # dust-synch * dust-synch 
        TDS = np.zeros((4, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1DS)
        TDS[0,:,:,:] = (EiEjd[self.MNi,self.MNh,:]*BiBjs[self.MNj,self.MNk,:] + Bis_Ejd[self.MNk,self.MNi,:]*Bis_Ejd[self.MNj,self.MNh,:])/(2*ell+1)
        # (2DS)
        TDS[1,:,:,:] = BiBjd[self.MNi,self.MNh,:]*EiEjs[self.MNj,self.MNk,:]/(2*ell+1) 
        # (3DS)
        TDS[2,:,:,:] = Eis_Ejd[self.MNk,self.MNi,:]*Bis_Bjd[self.MNj,self.MNh,:]/(2*ell+1)
        # (4DS)
        TDS[3,:,:,:] = Bis_Bjd[self.MNk,self.MNi,:]*Eis_Ejd[self.MNj,self.MNh,:]/(2*ell+1)
        TDS = np.moveaxis(bin_cov_matrix(TDS, binInfo), 3, 1)
        return To, Tcmb, Ts, Td, TSD, TDS


    def cov_elements_cross_o(self,Eis_Ejo, Bis_Bjo, Eis_Bjo, Bis_Ejo,
                            Eid_Ejo, Bid_Bjo, Eid_Bjo, Bid_Ejo,
                            binInfo, lmax):  
            
        ell=np.arange(0, lmax+1, 1)
        ##################### remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        # synch * observed 
        Ts_o = np.zeros((4,self.Nbands*(self.Nbands-1),self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1so)
        Ts_o[0,:,:,:] = (Eis_Ejo[self.MNi,self.MNh,:]*Bis_Bjo[self.MNj,self.MNk,:] + Eis_Bjo[self.MNi,self.MNk,:]*Bis_Ejo[self.MNj,self.MNh,:])/(2*ell+1)
        # (1so*)
        Ts_o[1,:,:,:] = (Eis_Ejo[self.MNh,self.MNi,:]*Bis_Bjo[self.MNk,self.MNj,:] + Eis_Bjo[self.MNh,self.MNj,:]*Bis_Ejo[self.MNk,self.MNi,:])/(2*ell+1)
        # (4so)
        Ts_o[2,:,:,:] = Bis_Bjo[self.MNi,self.MNk,:]*Eis_Ejo[self.MNj,self.MNh,:]/(2*ell+1) 
        # (4so*)
        Ts_o[3,:,:,:] = Bis_Bjo[self.MNh,self.MNj,:]*Eis_Ejo[self.MNk,self.MNi,:]/(2*ell+1)
        Ts_o = np.moveaxis(bin_cov_matrix(Ts_o, binInfo), 3, 1)
        ##################### remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        # dust * observed 
        Td_o = np.zeros((4,self.Nbands*(self.Nbands-1),self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1do)
        Td_o[0,:,:,:] = (Eid_Ejo[self.MNi,self.MNh,:]*Bid_Bjo[self.MNj,self.MNk,:] + Eid_Bjo[self.MNi,self.MNk,:]*Bid_Ejo[self.MNj,self.MNh,:])/(2*ell+1)
        # (1do*)
        Td_o[1,:,:,:] = (Eid_Ejo[self.MNh,self.MNi,:]*Bid_Bjo[self.MNk,self.MNj,:] + Eid_Bjo[self.MNh,self.MNj,:]*Bid_Ejo[self.MNk,self.MNi,:])/(2*ell+1)
        # (4do)
        Td_o[2,:,:,:] = Bid_Bjo[self.MNi,self.MNk,:]*Eid_Ejo[self.MNj,self.MNh,:]/(2*ell+1)
        # (4do*)
        Td_o[3,:,:,:] = Bid_Bjo[self.MNh,self.MNj,:]*Eid_Ejo[self.MNk,self.MNi,:]/(2*ell+1)
        Td_o = np.moveaxis(bin_cov_matrix(Td_o, binInfo), 3, 1)
        ##################### remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        # synch-dust * observed 
        TSD_o = np.zeros((4,self.Nbands*(self.Nbands-1),self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1SDo)
        TSD_o[0,:,:,:] = (Eis_Ejo[self.MNi,self.MNh,:]*Bid_Bjo[self.MNj,self.MNk,:] + Eis_Bjo[self.MNi,self.MNk,:]*Bid_Ejo[self.MNj,self.MNh,:])/(2*ell+1)
        # (1SDo*)
        TSD_o[1,:,:,:] = (Eis_Ejo[self.MNh,self.MNi,:]*Bid_Bjo[self.MNk,self.MNj,:] + Eis_Bjo[self.MNh,self.MNj,:]*Bid_Ejo[self.MNk,self.MNi,:])/(2*ell+1)
        # (4SDo)
        TSD_o[2,:,:,:] = Bis_Bjo[self.MNi,self.MNk,:]*Eid_Ejo[self.MNj,self.MNh,:]/(2*ell+1)
        # (4SDo*)
        TSD_o[3,:,:,:] = Bis_Bjo[self.MNh,self.MNj,:]*Eid_Ejo[self.MNk,self.MNi,:]/(2*ell+1)
        TSD_o = np.moveaxis(bin_cov_matrix(TSD_o, binInfo), 3, 1)
        ##################### remove EB from all except T0 and T1 (only T0 T1 T6 T7 left)
        # dust-synch * observed  
        TDS_o = np.zeros((4, self.Nbands*(self.Nbands-1),self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1DSo)
        TDS_o[0,:,:,:] = (Eid_Ejo[self.MNi,self.MNh,:]*Bis_Bjo[self.MNj,self.MNk,:] + Eid_Bjo[self.MNi,self.MNk,:]*Bis_Ejo[self.MNj,self.MNh,:])/(2*ell+1)
        # (1DSo*)
        TDS_o[1,:,:,:] = (Eid_Ejo[self.MNh,self.MNi,:]*Bis_Bjo[self.MNk,self.MNj,:] + Eid_Bjo[self.MNh,self.MNj,:]*Bis_Ejo[self.MNk,self.MNi,:])/(2*ell+1)
        # (4DSo)
        TDS_o[2,:,:,:] = Bid_Bjo[self.MNi,self.MNk,:]*Eis_Ejo[self.MNj,self.MNh,:]/(2*ell+1)
        # (4DSo*)
        TDS_o[3,:,:,:] = Bid_Bjo[self.MNh,self.MNj,:]*Eis_Ejo[self.MNk,self.MNi,:]/(2*ell+1)
        TDS_o = np.moveaxis(bin_cov_matrix(TDS_o, binInfo), 3, 1)
        #final matrix structure (modes, Nbins, N, N)
        return Ts_o, Td_o, TSD_o, TDS_o


    def cov_elements_cross_fg(self,EiEjs, BiBjs, EiBjs, BiEjs,
                            EiEjd, BiBjd, EiBjd, BiEjd,
                            Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd,
                            binInfo, lmax):  
            
        ell=np.arange(0, lmax+1,1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # synch * dust 
        Ts_d=np.zeros((8, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1sd)
        Ts_d[0,:,:,:] = (Eis_Ejd[self.MNi,self.MNh,:]*Bis_Bjd[self.MNj,self.MNk,:] + Eis_Bjd[self.MNi,self.MNk,:]*Bis_Ejd[self.MNj,self.MNh,:])/(2*ell+1)
        # (1sd*)
        Ts_d[1,:,:,:] = (Eis_Ejd[self.MNh,self.MNi,:]*Bis_Bjd[self.MNk,self.MNj,:] + Eis_Bjd[self.MNh,self.MNj,:]*Bis_Ejd[self.MNk,self.MNi,:])/(2*ell+1)
        # (2sd)
        Ts_d[2,:,:,:] = Bis_Bjd[self.MNi,self.MNh,:]*Eis_Ejd[self.MNj,self.MNk,:]/(2*ell+1)
        # (2sd*)
        Ts_d[3,:,:,:] = Bis_Bjd[self.MNh,self.MNi,:]*Eis_Ejd[self.MNk,self.MNj,:]/(2*ell+1)
        # (3sd) corregido
        Ts_d[4,:,:,:] = Eis_Ejd[self.MNi,self.MNk,:]*Bis_Bjd[self.MNj,self.MNh,:]/(2*ell+1)
        # (3sd*) corregido
        Ts_d[5,:,:,:] = Eis_Ejd[self.MNh,self.MNj,:]*Bis_Bjd[self.MNk,self.MNi,:]/(2*ell+1)
        # (4sd) corregido
        Ts_d[6,:,:,:] = Bis_Bjd[self.MNi,self.MNk,:]*Eis_Ejd[self.MNj,self.MNh,:]/(2*ell+1)
        # (4sd*) corregido
        Ts_d[7,:,:,:] = Bis_Bjd[self.MNh,self.MNj,:]*Eis_Ejd[self.MNk,self.MNi,:]/(2*ell+1)
        Ts_d = np.moveaxis(bin_cov_matrix(Ts_d, binInfo), 3, 1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # synch-dust * dust-synch
        TSD_DS = np.zeros((8, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1SDDS)
        TSD_DS[0,:,:,:] = (Eis_Ejd[self.MNi,self.MNh,:]*Bis_Bjd[self.MNk,self.MNj,:] + EiBjs[self.MNi,self.MNk,:]*BiEjd[self.MNj,self.MNh,:])/(2*ell+1)
        # (1SDDS*)
        TSD_DS[1,:,:,:] = (Eis_Ejd[self.MNh,self.MNi,:]*Bis_Bjd[self.MNj,self.MNk,:] + EiBjd[self.MNi,self.MNk,:]*BiEjs[self.MNj,self.MNh,:])/(2*ell+1)
        # (2SDDS)
        TSD_DS[2,:,:,:] = Bis_Bjd[self.MNi,self.MNh,:]*Eis_Ejd[self.MNk,self.MNj,:]/(2*ell+1)
        # (2SDDS*)
        TSD_DS[3,:,:,:] = Bis_Bjd[self.MNh,self.MNi,:]*Eis_Ejd[self.MNj,self.MNk,:]/(2*ell+1)
        # (3SDDS)
        TSD_DS[4,:,:,:] = EiEjs[self.MNi,self.MNk,:]*BiBjd[self.MNj,self.MNh,:]/(2*ell+1)
        # (3SDDS*)
        TSD_DS[5,:,:,:] = BiBjd[self.MNi,self.MNk,:]*EiEjs[self.MNj,self.MNh,:]/(2*ell+1)
        # (4SDDS)
        TSD_DS[6,:,:,:] = BiBjs[self.MNi,self.MNk,:]*EiEjd[self.MNj,self.MNh,:]/(2*ell+1)
        # (4SDDS*)
        TSD_DS[7,:,:,:] = EiEjd[self.MNi,self.MNk,:]*BiBjs[self.MNj,self.MNh,:]/(2*ell+1)
        TSD_DS = np.moveaxis(bin_cov_matrix(TSD_DS, binInfo), 3, 1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # synch * synch-dust 
        Ts_SD = np.zeros((8, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1sSD)
        Ts_SD[0,:,:,:] = (EiEjs[self.MNi,self.MNh,:]*Bis_Bjd[self.MNj,self.MNk,:] + Eis_Bjd[self.MNi,self.MNk,:]*BiEjs[self.MNj,self.MNh,:])/(2*ell+1)
        # (1sSD*)
        Ts_SD[1,:,:,:] = (EiEjs[self.MNh,self.MNi,:]*Bis_Bjd[self.MNk,self.MNj,:] + Eis_Bjd[self.MNh,self.MNj,:]*BiEjs[self.MNk,self.MNi,:])/(2*ell+1)
        # (2sSD)
        Ts_SD[2,:,:,:] = Eis_Ejd[self.MNi,self.MNk,:]*BiBjs[self.MNj,self.MNh,:]/(2*ell+1)
        # (2sSD*)
        Ts_SD[3,:,:,:] = Eis_Ejd[self.MNh,self.MNj,:]*BiBjs[self.MNk,self.MNi,:]/(2*ell+1)
        # (3sSD)
        Ts_SD[4,:,:,:] = Bis_Bjd[self.MNi,self.MNk,:]*EiEjs[self.MNj,self.MNh,:]/(2*ell+1)
        # (3sSD*)
        Ts_SD[5,:,:,:] = Bis_Bjd[self.MNh,self.MNj,:]*EiEjs[self.MNk,self.MNi,:]/(2*ell+1)
        # (4sSD)
        Ts_SD[6,:,:,:] = BiBjs[self.MNi,self.MNh,:]*Eis_Ejd[self.MNj,self.MNk,:]/(2*ell+1)
        # (4sSD*)
        Ts_SD[7,:,:,:] = BiBjs[self.MNh,self.MNi,:]*Eis_Ejd[self.MNk,self.MNj,:]/(2*ell+1)
        Ts_SD = np.moveaxis(bin_cov_matrix(Ts_SD, binInfo), 3, 1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # synch * dust-synch 
        Ts_DS=np.zeros((8, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1sDS)
        Ts_DS[0,:,:,:] = (Eis_Ejd[self.MNi,self.MNh,:]*BiBjs[self.MNj,self.MNk,:] + EiBjs[self.MNi,self.MNk,:]*Bis_Ejd[self.MNj,self.MNh,:])/(2*ell+1)
        # (1sDS*)
        Ts_DS[1,:,:,:] = (Eis_Ejd[self.MNh,self.MNi,:]*BiBjs[self.MNk,self.MNj,:] + EiBjs[self.MNh,self.MNj,:]*Bis_Ejd[self.MNk,self.MNi,:])/(2*ell+1)
        # (2sDS)
        Ts_DS[2,:,:,:] = EiEjs[self.MNi,self.MNk,:]*Bis_Bjd[self.MNj,self.MNh,:]/(2*ell+1)
        # (2sDS*)
        Ts_DS[3,:,:,:] = EiEjs[self.MNh,self.MNj,:]*Bis_Bjd[self.MNk,self.MNi,:]/(2*ell+1)
        # (3sDS)
        Ts_DS[4,:,:,:] = BiBjs[self.MNi,self.MNk,:]*Eis_Ejd[self.MNj,self.MNh,:]/(2*ell+1)
        # (3sDS*)
        Ts_DS[5,:,:,:] = BiBjs[self.MNh,self.MNj,:]*Eis_Ejd[self.MNk,self.MNi,:]/(2*ell+1)
        # (4sDS) 
        Ts_DS[6,:,:,:] = Bis_Bjd[self.MNi,self.MNh,:]*EiEjs[self.MNj,self.MNk,:]/(2*ell+1)
        # (4sDS*) 
        Ts_DS[7,:,:,:] = Bis_Bjd[self.MNh,self.MNi,:]*EiEjs[self.MNk,self.MNj,:]/(2*ell+1)
        Ts_DS = np.moveaxis(bin_cov_matrix(Ts_DS, binInfo), 3, 1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # dust * synch-dust 
        Td_SD=np.zeros((8, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1dSD)
        Td_SD[0,:,:,:] = (Eis_Ejd[self.MNh,self.MNi,:]*BiBjd[self.MNj,self.MNk,:] + EiBjd[self.MNi,self.MNk,:]*Eis_Bjd[self.MNh,self.MNj,:])/(2*ell+1)
        # (1dSD*)
        Td_SD[1,:,:,:] = (Eis_Ejd[self.MNi,self.MNh,:]*BiBjd[self.MNk,self.MNj,:] + EiBjd[self.MNh,self.MNj,:]*Eis_Bjd[self.MNi,self.MNk,:])/(2*ell+1)
        # (2dSD)
        Td_SD[2,:,:,:] = EiEjd[self.MNi,self.MNk,:]*Bis_Bjd[self.MNh,self.MNj,:]/(2*ell+1)
        # (2dSD*)
        Td_SD[3,:,:,:] = EiEjd[self.MNh,self.MNj,:]*Bis_Bjd[self.MNi,self.MNk,:]/(2*ell+1)
        # (3dSD)
        Td_SD[4,:,:,:] = BiBjd[self.MNi,self.MNk,:]*Eis_Ejd[self.MNh,self.MNj,:]/(2*ell+1)
        # (3dSD*)
        Td_SD[5,:,:,:] = BiBjd[self.MNh,self.MNj,:]*Eis_Ejd[self.MNi,self.MNk,:]/(2*ell+1)
        # (4dSD)
        Td_SD[6,:,:,:] = Bis_Bjd[self.MNh,self.MNi,:]*EiEjd[self.MNj,self.MNk,:]/(2*ell+1)
        # (4dSD*)
        Td_SD[7,:,:,:] = Bis_Bjd[self.MNi,self.MNh,:]*EiEjd[self.MNk,self.MNj,:]/(2*ell+1)
        Td_SD = np.moveaxis(bin_cov_matrix(Td_SD, binInfo), 3, 1)
        ##################### remove EB from T2, T3, T4, T5, T6, T7
        # dust * dust-synch 
        Td_DS=np.zeros((8, self.Nbands*(self.Nbands-1), self.Nbands*(self.Nbands-1), lmax+1), dtype=self.dt)
        # (1dDS)
        Td_DS[0,:,:,:] = (EiEjd[self.MNi,self.MNh,:]*Bis_Bjd[self.MNk,self.MNj,:] + Bis_Ejd[self.MNk,self.MNi,:]*BiEjd[self.MNj,self.MNh,:])/(2*ell+1)
        # (1dDS*)
        Td_DS[1,:,:,:] = (EiEjd[self.MNh,self.MNi,:]*Bis_Bjd[self.MNj,self.MNk,:] + Bis_Ejd[self.MNj,self.MNh,:]*BiEjd[self.MNk,self.MNi,:])/(2*ell+1)
        # (2dDS)
        Td_DS[2,:,:,:] = Eis_Ejd[self.MNk,self.MNi,:]*BiBjd[self.MNj,self.MNh,:]/(2*ell+1)
        # (2dDS*)
        Td_DS[3,:,:,:] = Eis_Ejd[self.MNj,self.MNh,:]*BiBjd[self.MNk,self.MNi,:]/(2*ell+1)
        # (3dDS)
        Td_DS[4,:,:,:] = Bis_Bjd[self.MNk,self.MNi,:]*EiEjd[self.MNj,self.MNh,:]/(2*ell+1)
        # (3dDS*)
        Td_DS[5,:,:,:] = Bis_Bjd[self.MNj,self.MNh,:]*EiEjd[self.MNk,self.MNi,:]/(2*ell+1)
        # (4dDS)
        Td_DS[6,:,:,:] = BiBjd[self.MNi,self.MNh,:]*Eis_Ejd[self.MNk,self.MNj,:]/(2*ell+1)
        # (4dDS*)
        Td_DS[7,:,:,:] = BiBjd[self.MNh,self.MNi,:]*Eis_Ejd[self.MNj,self.MNk,:]/(2*ell+1)
        Td_DS = np.moveaxis(bin_cov_matrix(Td_DS, binInfo), 3, 1)
        #final matrix structure (modes, Nbins, N, N)
        return Ts_d, TSD_DS, Ts_SD, Ts_DS, Td_SD, Td_DS


    def calculate(self,idx):
        try:
            cl_o_o,cl_d_o,cl_d_d,cl_s_d,cl_s_s,cl_s_o =self.spec.get_spectra(idx)
        except TypeError:
            self.spec.compute(idx)
            cl_o_o,cl_d_o,cl_d_d,cl_s_d,cl_s_s,cl_s_o =self.spec.get_spectra(idx)

        EEo_ij_b   = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        BBo_ij_b   = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        EBo_ij_b   = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        EBd_ij_b   = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        EBs_ij_b   = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        EsBd_ij_b  = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        EdBs_ij_b  = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        EEcmb_ij_b = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        BBcmb_ij_b = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt) 
        # for the covariance 
        # observed * observed
        EiEj_o = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); BiBj_o = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        EiBj_o = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); BiEj_o = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        # dust * dust
        EiEj_d = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); BiBj_d = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        EiBj_d = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); BiEj_d = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        # synch * synch
        EiEj_s = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); BiBj_s = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        EiBj_s = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); BiEj_s = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        # synch * dust
        Eis_Ejd = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); Bis_Bjd = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        Eis_Bjd = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); Bis_Ejd = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        # dust * observed
        Eid_Ejo = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); Bid_Bjo = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        Eid_Bjo = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); Bid_Ejo = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        # synch * observed
        Eis_Ejo = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); Bis_Bjo = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)
        Eis_Bjo = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt); Bis_Ejo = np.zeros((self.Nbands, self.Nbands, self.bmax+1), dtype=self.dt)

        for ii in range(0, self.Nbands, 1):
            fwhm_i = self.inst[self.bands[ii]]['fwhm']
            pos_i  = self.inst[self.bands[ii]]['idx']
            for jj in range(0, self.Nbands,1):
                fwhm_j = self.inst[self.bands[jj]]['fwhm']
                pos_j  = self.inst[self.bands[jj]]['idx']
                
                # for the fit
                EEo_ij_b[ii,jj,:]   = cl_o_o[pos_i, pos_j, 0, :self.bmax+1]
                BBo_ij_b[ii,jj,:]   = cl_o_o[pos_i, pos_j, 1, :self.bmax+1]
                EBo_ij_b[ii,jj,:]   = cl_o_o[pos_i, pos_j, 2, :self.bmax+1]
                EBd_ij_b[ii,jj,:]   = cl_d_d[pos_i, pos_j, 2, :self.bmax+1]
                EBs_ij_b[ii,jj,:]   = cl_s_s[pos_i, pos_j, 2, :self.bmax+1]
                EsBd_ij_b[ii,jj,:]  = cl_s_d[pos_i, pos_j, 2, :self.bmax+1]
                EdBs_ij_b[ii,jj,:]  = cl_s_d[pos_j, pos_i, 3, :self.bmax+1]
                EEcmb_ij_b[ii,jj,:] = self.convolveCls_gaussBeams_pwf(self.cmb_cls[1,:], fwhm_i, fwhm_j, self.bmax)
                BBcmb_ij_b[ii,jj,:] = self.convolveCls_gaussBeams_pwf(self.cmb_cls[2,:], fwhm_i, fwhm_j, self.bmax)

                # for the covariance
                # observed * observed
                EiEj_o[ii,jj,:] = cl_o_o[pos_i,pos_j,0,:self.bmax+1]
                BiBj_o[ii,jj,:] = cl_o_o[pos_i,pos_j,1,:self.bmax+1]
                EiBj_o[ii,jj,:] = cl_o_o[pos_i,pos_j,2,:self.bmax+1]
                BiEj_o[ii,jj,:] = cl_o_o[pos_j,pos_i,2,:self.bmax+1]
                # foregrounds * foregrounds
                EiEj_d[ii,jj,:]  = cl_d_d[pos_i,pos_j,0,:self.bmax+1]
                BiBj_d[ii,jj,:]  = cl_d_d[pos_i,pos_j,1,:self.bmax+1]
                EiBj_d[ii,jj,:]  = cl_d_d[pos_i,pos_j,2,:self.bmax+1]
                BiEj_d[ii,jj,:]  = cl_d_d[pos_j,pos_i,2,:self.bmax+1]
                EiEj_s[ii,jj,:]  = cl_s_s[pos_i,pos_j,0,:self.bmax+1]
                BiBj_s[ii,jj,:]  = cl_s_s[pos_i,pos_j,1,:self.bmax+1]
                EiBj_s[ii,jj,:]  = cl_s_s[pos_i,pos_j,2,:self.bmax+1]
                BiEj_s[ii,jj,:]  = cl_s_s[pos_j,pos_i,2,:self.bmax+1]
                Eis_Ejd[ii,jj,:] = cl_s_d[pos_i,pos_j,0,:self.bmax+1]
                Bis_Bjd[ii,jj,:] = cl_s_d[pos_i,pos_j,1,:self.bmax+1]
                Eis_Bjd[ii,jj,:] = cl_s_d[pos_i,pos_j,2,:self.bmax+1]
                Bis_Ejd[ii,jj,:] = cl_s_d[pos_i,pos_j,3,:self.bmax+1]
                # foregrounds * observed
                Eid_Ejo[ii,jj,:] = cl_d_o[pos_i,pos_j,0,:self.bmax+1]
                Bid_Bjo[ii,jj,:] = cl_d_o[pos_i,pos_j,1,:self.bmax+1]
                Eid_Bjo[ii,jj,:] = cl_d_o[pos_i,pos_j,2,:self.bmax+1]
                Bid_Ejo[ii,jj,:] = cl_d_o[pos_i,pos_j,3,:self.bmax+1]
                Eis_Ejo[ii,jj,:] = cl_s_o[pos_i,pos_j,0,:self.bmax+1]
                Bis_Bjo[ii,jj,:] = cl_s_o[pos_i,pos_j,1,:self.bmax+1]
                Eis_Bjo[ii,jj,:] = cl_s_o[pos_i,pos_j,2,:self.bmax+1]
                Bis_Ejo[ii,jj,:] = cl_s_o[pos_i,pos_j,3,:self.bmax+1]

        del cl_o_o, cl_d_d, cl_d_o, cl_s_s, cl_s_o, cl_s_d # free memory
        # bin only once at the end
        EEo_ij_b   = bin_spec_matrix(EEo_ij_b, self.bin_conf)
        BBo_ij_b   = bin_spec_matrix(BBo_ij_b, self.bin_conf)
        EBo_ij_b   = bin_spec_matrix(EBo_ij_b, self.bin_conf)
        EBd_ij_b   = bin_spec_matrix(EBd_ij_b, self.bin_conf)
        EBs_ij_b   = bin_spec_matrix(EBs_ij_b, self.bin_conf)
        EsBd_ij_b  = bin_spec_matrix(EsBd_ij_b, self.bin_conf)
        EdBs_ij_b  = bin_spec_matrix(EdBs_ij_b, self.bin_conf)
        EEcmb_ij_b = bin_spec_matrix(EEcmb_ij_b, self.bin_conf)
        BBcmb_ij_b = bin_spec_matrix(BBcmb_ij_b, self.bin_conf)

        
        To, Tcmb, Ts, Td, TSD, TDS = self.cov_elements_auto(EiEj_o, BiBj_o, EiBj_o, BiEj_o,
                                                    EiEj_s, BiBj_s, EiBj_s, BiEj_s,
                                                    EiEj_d, BiBj_d, EiBj_d, BiEj_d,
                                                    Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd,
                                                    self.std_i, self.std_j, self.std_h, self.std_k, self.cmb_cls, self.bin_conf, self.bmax)

        Ts_o, Td_o, TSD_o, TDS_o = self.cov_elements_cross_o(Eis_Ejo, Bis_Bjo, Eis_Bjo, Bis_Ejo,
                                                        Eid_Ejo, Bid_Bjo, Eid_Bjo, Bid_Ejo, self.bin_conf, self.bmax)

        Ts_d, TSD_DS, Ts_SD, Ts_DS, Td_SD, Td_DS = self.cov_elements_cross_fg(EiEj_s, BiBj_s, EiBj_s, BiEj_s,
                                                                        EiEj_d, BiBj_d, EiBj_d, BiEj_d,
                                                                        Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd, self.bin_conf, self.bmax)
        
        # free memory
        del EiEj_o, BiBj_o, EiBj_o, BiEj_o
        del EiEj_d, BiBj_d, EiBj_d, BiEj_d, EiEj_s, BiBj_s, EiBj_s, BiEj_s, Eis_Ejd, Bis_Bjd, Eis_Bjd, Bis_Ejd
        del Eid_Ejo, Bid_Bjo, Eid_Bjo, Bid_Ejo, Eis_Ejo, Bis_Bjo, Eis_Bjo, Bis_Ejo


        ang_list = [ np.concatenate((np.repeat(1, self.ExtParam-1), np.zeros(self.Nvar-self.ExtParam+1,dtype=self.dt))) ]
        cov_list = [ np.zeros((self.Nvar, self.Nvar),dtype=self.dt) ]
        std_list = [ np.zeros(self.Nvar, dtype=self.dt) ]

        converged = False
        niter = 0
        while not converged:
            # (Nbins, N(N-1), N(N-1))
            cov = self.build_cov(To, Tcmb, Ts, Td, TSD, TDS, Ts_o, Td_o, TSD_o, TDS_o, Ts_d, TSD_DS, Ts_SD, Ts_DS, Td_SD, Td_DS,
                                ang_list[niter])
            invcov = np.linalg.inv(cov/self.fsky)
            # build terms of system matrix
            B_ijhk = np.zeros((self.Nbands,self.Nbands, self.Nbands,self.Nbands), dtype=self.dt)
            E_ijhk = np.zeros((self.Nbands,self.Nbands, self.Nbands,self.Nbands), dtype=self.dt)
            I_ijhk = np.zeros((self.Nbands,self.Nbands, self.Nbands,self.Nbands), dtype=self.dt)
            #################
            D_ij       = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)
            H_ij       = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)
            nu_ij      = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)
            pi_ij      = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)
            rho_ij     = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)
            sigma_ij   = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)
            tau_ij     = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)
            varphi_ij  = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)
            phi_ij     = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)
            psi_ij     = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)  
            OMEGA_ij   = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)  
            omega_ij   = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)  
            ene_ij     = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)  
            epsilon_ij = np.zeros((self.Nbands,self.Nbands), dtype=self.dt)  
            #################
            A  = 0; C   = 0; F     = 0; G      = 0; R  = 0; S = 0; T = 0; J = 0; M     = 0; K = 0 
            O  = 0; P   = 0; Q     = 0; V      = 0; W  = 0; X = 0; Y = 0; Z = 0; DELTA = 0 
            xi = 0; eta = 0; theta = 0; LAMBDA = 0; mu = 0; L = 0; U = 0; N = 0; delta = 0
            for MN_pair in self.MNidx:
                ii,jj,hh,kk,mm,nn = self.get_index(MN_pair)
                B_ijhk[ii,jj,hh,kk] = np.sum(BBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*BBo_ij_b[hh,kk,:])
                E_ijhk[ii,jj,hh,kk] = np.sum(EEo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EEo_ij_b[hh,kk,:])
                I_ijhk[ii,jj,hh,kk] = np.sum(BBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EEo_ij_b[hh,kk,:])
                #################
                D_ij[ii,jj]       += np.sum(EEo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBo_ij_b[hh,kk,:])
                H_ij[ii,jj]       += np.sum(BBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBo_ij_b[hh,kk,:])
                nu_ij[ii,jj]      += np.sum(EEo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBs_ij_b[hh,kk,:])
                pi_ij[ii,jj]      += np.sum(EEo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EsBd_ij_b[hh,kk,:]) 
                rho_ij[ii,jj]     += np.sum(EEo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EdBs_ij_b[hh,kk,:])
                sigma_ij[ii,jj]   += np.sum(EEo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBd_ij_b[hh,kk,:])
                tau_ij[ii,jj]     += np.sum(EEo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EEcmb_ij_b[hh,kk,:]) 
                varphi_ij[ii,jj]  += np.sum(EEo_ij_b[ii,jj,:]*invcov[:,mm,nn]*BBcmb_ij_b[hh,kk,:])
                phi_ij[ii,jj]     += np.sum(BBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EsBd_ij_b[hh,kk,:]) 
                psi_ij[ii,jj]     += np.sum(BBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBs_ij_b[hh,kk,:])
                OMEGA_ij[ii,jj]   += np.sum(BBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EdBs_ij_b[hh,kk,:])
                omega_ij[ii,jj]   += np.sum(BBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBd_ij_b[hh,kk,:])
                ene_ij[ii,jj]     += np.sum(BBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EEcmb_ij_b[hh,kk,:]) 
                epsilon_ij[ii,jj] += np.sum(BBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*BBcmb_ij_b[hh,kk,:])
                #################
                A      += np.sum(EBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBo_ij_b[hh,kk,:]) 
                C      += np.sum(EEcmb_ij_b[ii,jj,:]*invcov[:,mm,nn]*BBcmb_ij_b[hh,kk,:])
                F      += np.sum(EEcmb_ij_b[ii,jj,:]*invcov[:,mm,nn]*EEcmb_ij_b[hh,kk,:])
                G      += np.sum(BBcmb_ij_b[ii,jj,:]*invcov[:,mm,nn]*BBcmb_ij_b[hh,kk,:])
                R      += np.sum(EBd_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBd_ij_b[hh,kk,:])
                S      += np.sum(EBs_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBs_ij_b[hh,kk,:])
                T      += np.sum(EsBd_ij_b[ii,jj,:]*invcov[:,mm,nn]*EsBd_ij_b[hh,kk,:])
                J      += np.sum(EBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBs_ij_b[hh,kk,:])
                M      += np.sum(EBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EdBs_ij_b[hh,kk,:])
                N      += np.sum(EBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBd_ij_b[hh,kk,:])
                U      += np.sum(EdBs_ij_b[ii,jj,:]*invcov[:,mm,nn]*EdBs_ij_b[hh,kk,:])
                L      += np.sum(EBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EsBd_ij_b[hh,kk,:])
                O      += np.sum(EBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*EEcmb_ij_b[hh,kk,:])
                P      += np.sum(EBo_ij_b[ii,jj,:]*invcov[:,mm,nn]*BBcmb_ij_b[hh,kk,:])
                Q      += np.sum(EBs_ij_b[ii,jj,:]*invcov[:,mm,nn]*EsBd_ij_b[hh,kk,:])
                V      += np.sum(EBs_ij_b[ii,jj,:]*invcov[:,mm,nn]*EdBs_ij_b[hh,kk,:])        
                W      += np.sum(EBs_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBd_ij_b[hh,kk,:])
                X      += np.sum(EBs_ij_b[ii,jj,:]*invcov[:,mm,nn]*EEcmb_ij_b[hh,kk,:])
                Y      += np.sum(EBs_ij_b[ii,jj,:]*invcov[:,mm,nn]*BBcmb_ij_b[hh,kk,:])
                Z      += np.sum(EsBd_ij_b[ii,jj,:]*invcov[:,mm,nn]*EdBs_ij_b[hh,kk,:])
                K      += np.sum(EdBs_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBd_ij_b[hh,kk,:])
                DELTA  += np.sum(EsBd_ij_b[ii,jj,:]*invcov[:,mm,nn]*EEcmb_ij_b[hh,kk,:])
                delta  += np.sum(EsBd_ij_b[ii,jj,:]*invcov[:,mm,nn]*BBcmb_ij_b[hh,kk,:])
                xi     += np.sum(EsBd_ij_b[ii,jj,:]*invcov[:,mm,nn]*EBd_ij_b[hh,kk,:])
                eta    += np.sum(EdBs_ij_b[ii,jj,:]*invcov[:,mm,nn]*EEcmb_ij_b[hh,kk,:])
                theta  += np.sum(EdBs_ij_b[ii,jj,:]*invcov[:,mm,nn]*BBcmb_ij_b[hh,kk,:])
                LAMBDA += np.sum(EBd_ij_b[ii,jj,:]*invcov[:,mm,nn]*EEcmb_ij_b[hh,kk,:])
                mu     += np.sum(EBd_ij_b[ii,jj,:]*invcov[:,mm,nn]*BBcmb_ij_b[hh,kk,:])

            # build system matrix and independent term
            sys_mat  = np.zeros((self.Nvar, self.Nvar),dtype=self.dt)
            ind_term = np.zeros(self.Nvar, dtype=self.dt)
            
            # variables ordered as As, Ad, Asd, beta, alpha_i
            sys_mat[0,0] = S      ; ind_term[0]  = J # As - As
            sys_mat[0,1] = W      ; sys_mat[1,0] = W # As - Ad
            sys_mat[0,2] = Q+V    ; sys_mat[2,0] = Q+V # As - Asd
            sys_mat[0,3] = 2*(X-Y); sys_mat[3,0] = 2*(X-Y) # As - beta

            sys_mat[1,1] = R            ; ind_term[1]  = N # Ad - Ad
            sys_mat[1,2] = K+xi         ; sys_mat[2,1] = K+xi # Ad - Asd
            sys_mat[1,3] = 2*(LAMBDA-mu); sys_mat[3,1] = 2*(LAMBDA-mu) # Ad - beta

            sys_mat[2,2] = T+U+2*Z                  ; ind_term[2] = M+L # Asd - Asd
            sys_mat[2,3] = 2*(DELTA-delta+eta-theta); sys_mat[3,2] = 2*(DELTA-delta+eta-theta) # Asd - beta

            sys_mat[3,3] = 4*(G+F-2*C); ind_term[3] = 2*(O-P)# beta - beta

            for aa in range(0, self.Nbands, 1):
                # As - alpha_i
                As_ai = np.sum(nu_ij[:,aa]) - np.sum(psi_ij[aa,:])
                sys_mat[0,aa+self.ExtParam] = 2*As_ai; sys_mat[aa+self.ExtParam,0] = 2*As_ai

                # Ad - alpha_i
                Ad_ai = np.sum(sigma_ij[:,aa]) - np.sum(omega_ij[aa,:])
                sys_mat[1,aa+self.ExtParam] = 2*Ad_ai; sys_mat[aa+self.ExtParam,1] = 2*Ad_ai

                # Asd - alpha_i
                Asd_ai = np.sum(pi_ij[:,aa]) + np.sum(rho_ij[:,aa]) - np.sum(phi_ij[aa,:]) - np.sum(OMEGA_ij[aa,:])
                sys_mat[2,aa+self.ExtParam] = 2*Asd_ai; sys_mat[aa+self.ExtParam,2] = 2*Asd_ai

                # beta - alpha_i
                b_a = np.sum(tau_ij[:,aa]) + np.sum(epsilon_ij[aa,:]) - np.sum(varphi_ij[:,aa]) - np.sum(ene_ij[aa,:])
                sys_mat[3,aa+self.ExtParam] = 4*b_a; sys_mat[aa+self.ExtParam,3] = 4*b_a

                # alpha_i
                ind_term[aa+self.ExtParam] = 2*(np.sum(D_ij[:,aa]) - np.sum(H_ij[aa,:]))
                for bb in range(0, self.Nbands, 1):
                    # alpha_i - alpha_j terms
                    aux1 = np.sum(E_ijhk[:, bb, :, aa]) + np.sum(E_ijhk[:, aa, :, bb])
                    aux2 = np.sum(B_ijhk[bb, :, aa, :]) + np.sum(B_ijhk[aa, :, bb, :])
                    aux3 = np.sum(I_ijhk[bb, :, :, aa]) + np.sum(I_ijhk[aa, :, :, bb])
                    sys_mat[aa+self.ExtParam,bb+self.ExtParam] = 2*( aux1 + aux2 - 2*aux3 )

            #solve Ax=B
            ang_now = np.linalg.solve(sys_mat, ind_term)
            cov_now = np.linalg.inv(sys_mat)
            std_now = np.sqrt(np.diagonal(cov_now))

            if np.any( np.isnan(std_now) ):
                #stop iterating 
                print('NaN in covariance')
                converged=True
            else:
                #evaluate convergence of the iterative calculation 
                #regulate tolerance depending on the sensitivity to angle measurement
                # tol= 0.00001 if np.min(std_now)*rad2arcmin >0.5 else 0.1
                # # use only the angles as convergence criterion, not amplitude
                # # use alpha + beta sum as convergence criterion
                # #difference with i-1
                # if niter == 0:
                #     c1vec = np.abs((ang_now[self.ExtParam-1]+ang_now[self.ExtParam:])-(ang_list[niter][self.ExtParam-1]+ang_list[niter][self.ExtParam:]))*rad2arcmin
                #     c1 = c1vec>=tol
                #     if np.sum(c1)<=1 or niter>self.niter_max:
                #         converged = True
                # else:
                #     #difference with i-2 
                #     c2vec = np.abs((ang_now[self.ExtParam-1]+ang_now[self.ExtParam:])-(ang_list[niter-1][self.ExtParam-1]+ang_list[niter-1][self.ExtParam:]))*rad2arcmin
                #     c2 = c2vec>=tol
                #     if np.sum(c2)<=1:
                #         converged = True

                tol= 0.5 if np.min(std_now)*rad2arcmin >0.5 else 0.1
                # use only the angles as convergence criterion, not amplitude
                # use alpha + beta sum as convergence criterion
                #difference with i-1
                c1 = np.abs((ang_now[self.ExtParam-1]+ang_now[self.ExtParam:])-(ang_list[niter][self.ExtParam-1]+ang_list[niter][self.ExtParam:]))*rad2arcmin>=tol
                if np.sum(c1)<=1 or niter>self.niter_max:
                    converged = True
                elif niter>0:
                    #difference with i-2 
                    c2 = np.abs((ang_now[self.ExtParam-1]+ang_now[self.ExtParam:])-(ang_list[niter-1][self.ExtParam-1]+ang_list[niter-1][self.ExtParam:]))*rad2arcmin>=tol
                    if np.sum(c2)<=1:
                        converged = True

            #store results
            ang_list.append(ang_now)
            cov_list.append(cov_now)
            std_list.append(std_now)
            niter += 1


        return ang_list,cov_list,std_list
        
    
    def estimate_angle(self, idx):
        ang_list, _,_ = self.calculate(idx)
        angs = np.rad2deg(ang_list[-1])
        result = {}
        for i, b in enumerate(self.bands):
            result[f"alpha_{b}"] = angs[i+4]
        result['beta'] = angs[3]
        return result

