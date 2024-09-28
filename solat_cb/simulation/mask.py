
class Mask:
    def __init__(self, libdir: str, nside: int, tele: str) -> None:
        """
        Initializes the Mask class for handling and generating sky masks.

        Parameters:
        nside (int): HEALPix resolution parameter.
        libdir (Optional[str], optional): Directory where the mask may be saved or loaded from. Defaults to None.
        """
        self.nside     = nside
        self.libdir    = os.path.join(libdir, "Mask")
        if mpi.rank == 0:
            os.makedirs(self.libdir, exist_ok=True)
        assert tele in ["LAT", "SAT","OVERLAP"], "telescope should be 'LAT' or 'SO'"
        self.tele = tele

        if self.tele == "OVERLAP":
            self.mask = None
        else:
            self.mask = self.__get_mask__()
        
        self.fsky = self.__calculate_fsky__()

    
    def __calculate_fsky__(self) -> float:
        """
        Calculate the sky fraction (fsky) from the mask.
        
        Returns:
        float: The fsky value.
        """
        if self.mask is None:
            return 0
        else:
            return np.mean(self.mask ** 2) ** 2 / np.mean(self.mask ** 4)

    def __get_mask__(self) -> np.ndarray:
        """
        Returns:
        np.ndarray: The mask as a NumPy array. If nhits is False, returns a binary mask; otherwise, returns the hit count map.
        """
        select = {'SAT':0, 'LAT':1}
        fname = os.path.join(self.libdir, f"mask_{self.tele}.fits")
        if os.path.isfile(fname):
            mask = hp.read_map(fname,select[self.tele])
        else:
            rmt_fname = 'https://figshare.com/ndownloader/files/49232491'
            download_file(rmt_fname, fname)
            mask = hp.read_map(fname,select[self.tele])

        nside = hp.npix2nside(len(mask))

        if nside != self.nside:
            mask = hp.ud_grade(mask, self.nside)
        
        return mask
    
    def __mul__(self, other):
        """
        Multiplies two Mask objects and returns a new Mask object.
        
        Parameters:
        other (Mask): Another Mask object to multiply with.
        
        Returns:
        Mask: A new Mask object with the combined mask and updated fsky.
        """
        assert self.nside == other.nside, "Masks must have the same nside"
        
        if self.mask is None or other.mask is None:
            raise ValueError("Cannot multiply masks when one of them is None")
        combined_mask = self.mask * other.mask
        
        # Create a new Mask object for the combined mask
        combined_mask_obj = Mask(libdir=self.libdir, nside=self.nside, tele="OVERLAP")
        combined_mask_obj.mask = combined_mask
        combined_mask_obj.fsky = np.mean(combined_mask ** 2) ** 2 / np.mean(combined_mask ** 4)
        
        return combined_mask_obj