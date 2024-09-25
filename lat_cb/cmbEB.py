import warnings
warnings.warn("cmb_temp.py is only used for Carlos' EB studies and will be removed in future.",stacklevel=2)
import os
import pickle 

fname = os.path.join('/tmp','all_cls_th.pkl')
if os.path.exists(fname):
    all_cls_th = pickle.load(open(fname,'rb'))
else:
    import camb
    ellmax = 6000
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.36, ombh2=0.02237, omch2=0.12, mnu=0.06, omk=0, tau=0.0544)
    pars.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0)
    pars.set_for_lmax(ellmax, lens_potential_accuracy=1)
    results = camb.get_results(pars)
    all_cls_th = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)['total']
    pickle.dump(all_cls_th,open(fname,'wb'))