

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
            return hp.read_map(fname, field=[0, 1])  
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
            hp.write_map(fname, QU, dtype=np.float64)
            return QU