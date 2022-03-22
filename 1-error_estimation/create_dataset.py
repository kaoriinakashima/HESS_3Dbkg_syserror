import numpy as np
import astropy.units as u
import yaml
import utils

from gammapy.data import DataStore
from gammapy.maps import MapAxis, WcsGeom, Map
from gammapy.datasets import MapDataset
from gammapy.makers import MapDatasetMaker, SafeMaskMaker, FoVBackgroundMaker
from gammapy.modeling.models import FoVBackgroundModel, Models
from gammapy.modeling import Fit
from gammapy.irf import Background3D

import warnings
warnings.filterwarnings('ignore')

idx=0

# loading general parameters
with open("../general_config.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
conf=cfg['conf']
hesseras= ['hess1', 'hess2']
muoneff_flag= cfg['muoneff_flag']

# defining the geometry for the datasets
energy_bins = np.logspace(-1, 2, cfg['N_ebins']+1)
axis = MapAxis.from_edges(energy_bins, unit="TeV", name="energy", interp="log")
maker = MapDatasetMaker()
maker_safe_mask = SafeMaskMaker(methods=['offset-max', 'bkg-peak'], offset_max=cfg['offset_cut'] * u.deg)

overall_problem = []
for hessera in hesseras[:1]:
    runlist = np.loadtxt(f'initial_runlist_{hessera}.txt')
    muoneff_path = f'{cfg["muoneff_path"]}/{hessera}/hess1_hess2/v01c_kaori_mueff'
    
    basedir = f'{cfg["FITS_PROD"]}/{hessera}/std_{conf}_fullEnclosure'
    ds = DataStore.from_dir(basedir, f'hdu-index-bg-latest-fov-radec.fits.gz', f'obs-index-bg-latest-fov-radec.fits.gz')
    observations = ds.get_observations(runlist[idx*200:(idx+1)*200])
    
    result_list = []
    runs_with_problems = []
    for obs in observations:
        try:
            # taking care of the correct bkg model (if it is muoneff or official model)
            # the official model is stored in the fits tables. For the muoneff model, we will just change the path of the bkg model
            if muoneff_flag:
                if obs.obs_info['MUONEFF'] > 0.085:
                    model_CD = 'B'
                elif obs.obs_info['MUONEFF'] >= 0.075:
                    model_CD = 'D'
                else:
                    model_CD = 'C'

                if obs.obs_id >= 100000:
                    run_number= f'{obs.obs_id}'
                else:
                    run_number= f'0{obs.obs_id}'
                filename = f'{muoneff_path}_{model_CD}/hess_bkg_3d_v01c_kaori_mueff_{model_CD}_norebin_fov_radec_{run_number}.fits.gz'
                obs.bkg = Background3D.read(filename, hdu='BACKGROUND')

            geom = WcsGeom.create(skydir=obs.pointing_radec, binsz=cfg['binsz'], width=cfg['width']* u.deg, frame="icrs", axes=[axis])        
            dataset = MapDataset.create(geom=geom)
            dataset = maker.run(dataset, obs)
            dataset = maker_safe_mask.run(dataset, obs)

            # create here mask_fit
            dataset.mask_fit = Map.from_geom(geom=geom, data=np.ones_like(dataset.counts.data).astype(bool))
            coord = utils.get_mask_fov(obs.pointing_radec.ra.deg, obs.pointing_radec.dec.deg, 5)
            if coord != 0: #this means that if there are regions to be masked
                for s in coord:
                    dataset.mask_fit &= ~dataset.counts.geom.region_mask(f"icrs;circle({s[0]}, {s[1]}, {s[2]})")

            bkg_model = FoVBackgroundModel(dataset_name=dataset.name)
            dataset.models = Models([bkg_model])
            dataset.background_model.spectral_model.tilt.frozen = False

            Fit().run(datasets=[dataset])

            if dataset.background_model.spectral_model.norm.value > 0:
                c = np.nansum((dataset.counts * dataset.mask_fit *dataset.mask_safe).data, axis=(1, 2))
                b = np.nansum((dataset.npred() * dataset.mask_fit *dataset.mask_safe).data, axis=(1, 2))
                result_list.append(np.concatenate([[obs.obs_id], c, b]))
            else:
                runs_with_problems.append(obs.obs_id)
        except:
            overall_problem.append(obs.obs_id)
    
    # dividing the result of hess1, because only one file is too big
    if hessera == 'hess1':
        np.savetxt(f'results/dataspectrum_muoneff_{hessera}_part{idx}.txt', np.asarray(result_list))
    else:
        np.savetxt(f'results/dataspectrum_muoneff_{hessera}.txt', np.asarray(result_list))
    np.savetxt(f'results/runs_with_problems_{hessera}_{idx}.txt', np.asarray(runs_with_problems))
np.savetxt(f'results/overall_problem.txt', np.asarray(overall_problem))