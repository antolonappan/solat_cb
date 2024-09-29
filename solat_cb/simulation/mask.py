# This file contains the Mask class for handling and generating sky masks.

# General imports
import os
import numpy as np
import healpy as hp
from pymaster import mask_apodization
# Local imports
from solat_cb import mpi
from solat_cb.data import SAT_MASK, LAT_MASK, CO_MASK, PS_MASK
from solat_cb.utils import Logger
class Mask:
    def __init__(self, 
                 libdir: str, 
                 nside: int, 
                 select: str, 
                 apo_scale: float = 0.0,
                 apo_method: str = 'C2',
                 verbose: bool=True) -> None:
        """
        Initializes the Mask class for handling and generating sky masks.

        Parameters:
        nside (int): HEALPix resolution parameter.
        libdir (Optional[str], optional): Directory where the mask may be saved or loaded from. Defaults to None.
        """
        self.logger = Logger(self.__class__.__name__,verbose)
        self.libdir = libdir
        self.nside = nside
        self.select = select
        self.apo_scale = apo_scale
        self.apo_method = apo_method
        self.mask = self.__load_mask__()
        self.fsky = self.__calc_fsky__()

    def __mask_obj__(self, select: str):
        match select:
            case "SAT":
                mask = SAT_MASK
            case "LAT":
                mask = LAT_MASK
            case "CO":
                mask = CO_MASK
            case "PS":
                mask = PS_MASK
            case _:
                raise ValueError(f"Invalid mask selection: {self.select}")
        return mask

    def __load_mask_healper__(self) -> np.ndarray:
        """
        Loads a mask from a file.

        Returns:
        np.ndarray: The mask array.
        """
        if 'x' in self.select:
            self.logger.log("Loading composite mask", level="info")
            masks = self.select.split('x')
            final_mask = np.ones(hp.nside2npix(self.nside))
            fsky = []
            for mask in masks:
                maskobj = self.__mask_obj__(mask)
                maskobj.directory = self.libdir
                smask = maskobj.data
                if hp.get_nside(smask) > self.nside:
                    self.logger.log(f"Downgrading mask {mask} resolution", level="info")
                else:
                    self.logger.log(f"Upgrading mask {mask} resolution", level="info")
                smask = hp.ud_grade(smask, self.nside)
                fsky.append(self.__calc_fsky__(smask))
                final_mask *= smask
            fskyb = sorted(set(fsky))[-2]
            fskyf = self.__calc_fsky__(final_mask)
            self.logger.log(f"Composite Mask {self.select}: fsky changed {fskyb:.2f} -> {fskyf:.2f}  ", level="info")
        else:
            mask = self.__mask_obj__(self.select)
            mask.directory = self.libdir
            final_mask = mask.data
            if hp.get_nside(final_mask) != self.nside:
                if hp.get_nside(final_mask) > self.nside:
                    self.logger.log(f"Downgrading mask {self.select} resolution", level="info")
                else:
                    self.logger.log(f"Upgrading mask {self.select} resolution", level="info")
                final_mask = hp.ud_grade(final_mask, self.nside)
        return np.array(final_mask)
    
    def __load_mask__(self) -> np.ndarray:
        """
        Loads a mask from a file.

        Returns:
        np.ndarray: The mask array.
        """
        mask = self.__load_mask_healper__()
        if self.apo_scale > 0:
            fskyb = self.__calc_fsky__(mask)
            self.logger.log(f"Apodizing mask: scale {self.apo_scale}: method: {self.apo_method}", level="info")
            mask = mask_apodization(mask, self.apo_scale, apotype=self.apo_method)
            fskya = self.__calc_fsky__(mask)
            self.logger.log(f"Apodizing changed the fsky {fskyb:.3f} -> {fskya:.3f}", level="info") 
        return mask

    def __calc_fsky__(self,mask=None) -> float:
        """
        Calculates the fraction of sky covered by the mask.

        Returns:
        float: The fraction of sky covered by the mask.
        """
        if mask is None:
            mask = self.mask
        return float(np.mean(mask ** 2) ** 2 / np.mean(mask ** 4))

    
