import pyximport

pyximport.install()
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import gammapy

# from gammapy.datasets import MapDataset
from gammapy.maps import Map
from astropy.coordinates import SkyCoord, Angle
from gammapy.modeling import Fit, Parameter, Parameters, Covariance
from gammapy.datasets import MapDataset #, MapDatasetNuisance
from gammapy.modeling.models import (
    PowerLawSpectralModel,
    create_crab_spectral_model,
    SkyModel,
    PointSpatialModel,
    ShellSpatialModel,
    GeneralizedGaussianSpatialModel,
    TemplateSpatialModel,
    LogParabolaSpectralModel,
    GaussianSpatialModel,
    DiskSpatialModel,
    PowerLawNormSpectralModel,
    Models,
    SpatialModel,
    FoVBackgroundModel,
)
from regions import CircleSkyRegion

import sys

sys.path.append(
    "/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/syserror_3d_bkgmodel/4-Fitting_nuisance_and_model_parameters"
)
from my_dataset_maps_19 import MapDatasetNuisance
from  my_fit_19 import Fit

name= 'test3'

main_path = '/home/vault/caph/mppi062h/repositories/syserror_3d_bkgmodel/2-source_dataset/GC_0.19'

hess = MapDataset.read(f'{main_path}/20220511_dataset002_hess1_muonflagTrue.fits')
hess2 = MapDataset.read(f'{main_path}/20220511_dataset002_hess2_muonflagTrue.fits')
hess.stack(hess2)

dataset_standard = hess
dataset_standard = dataset_standard.cutout(
    position=dataset_standard.geoms["geom"].center_skydir, width=6 * u.deg
)

from gammapy.catalog import SourceCatalogHGPS, SourceCatalogGammaCat, SourceCatalog4FGL
from gammapy.modeling.models import ExpCutoffPowerLawSpectralModel, GaussianSpatialModel, ShellSpatialModel

HGPS, FGL4 = SourceCatalogHGPS(), SourceCatalog4FGL() 
catalog = HGPS
mask1 = np.abs(catalog.table['GLAT']) < 2.6
mask2 = (catalog.table['GLON'] < 2.6) + (catalog.table['GLON'] > 357.4)
mask = mask1 & mask2
HGPS_models = catalog[mask].to_models()

wun_map = '/home/saturn/caph/mppi043h/diffusiontemplate/imp_pcut_v3.fits'
print(wun_map)
diff_map = Map.read(wun_map)
if wun_map == '/home/saturn/caph/mppi043h/diffusiontemplate/imp_v2.fits':
    diff_map.geom.axes['energy'].name = 'energy_true'
diff_name = f'diffuse_model.fits'
diff_map.write(diff_name, overwrite=True)
diff= SkyModel(spectral_model=PowerLawNormSpectralModel(),
               spatial_model=TemplateSpatialModel(diff_map, normalize=False, filename=diff_name),
               name='diff-emission')
#diff.parameters["tilt"].frozen = False


# assining the models to the dataset 
initial_model_hess = HGPS_models
initial_model_hess.append(diff)
initial_model_hess['HESS J1745-290'].spatial_model=GaussianSpatialModel(lon_0=359.945*u.deg, lat_0=-0.044*u.deg, sigma=0.07*u.deg, frame='galactic')


# these sources are within the mask fit and therefore does not need to be fitted
frozen_sources= ['HESS J1746-308','HESS J1745-303']
for s in frozen_sources:
    initial_model_hess[s].parameters.freeze_all()

bkg_model = FoVBackgroundModel(dataset_name=dataset_standard.name)
bkg_model.parameters["tilt"].frozen = False
initial_model_hess.append(bkg_model)
dataset_standard.models = initial_model_hess

dataset_standard.mask_fit = Map.from_geom(geom=dataset_standard.counts.geom, data=np.ones_like(dataset_standard.counts.data).astype(bool))   
dataset_standard.mask_fit &= ~dataset_standard.counts.geom.region_mask(f"galactic;circle(358.71, -0.64, 0.7)")

fit_standarad = Fit(store_trace=False)
result_standarad = fit_standarad.run([dataset_standard])

downsampling_factor = 100
# read the real sys amplitude here ... 
sysamplitude = abs(np.loadtxt('sysamplitude.txt'))/100 # this is because the sys is now in units of % of bkg

nuisance_mask_helper = np.array(sysamplitude)>0
print('nui_mask:',nuisance_mask_helper)
nuisance_mask = Map.from_geom(dataset_standard.geoms["geom"].downsample(downsampling_factor),
                             dtype=bool)
for e, n in enumerate(nuisance_mask_helper):
    nuisance_mask.data[e,:,:] = n

ndim_3D_nui = sum(nuisance_mask.data.flatten())
ndim_spatial_nui_1D = (
    dataset_standard.geoms["geom"]
    .downsample(downsampling_factor)
    .data_shape[1]
)
ndim_spatial_nui = ndim_spatial_nui_1D ** 2
ndim_spectral_nui = len(np.where(nuisance_mask_helper)[0])

print(f"Number of nuisance parameters: {ndim_3D_nui}")
print(
    "at energy: {:.2} : {:.2}".format(
        dataset_standard.geoms["geom"].axes[0].edges[np.where(nuisance_mask_helper)[0][0]],
        dataset_standard.geoms["geom"].axes[0].edges[np.where(nuisance_mask_helper)[0][-1]+1 ],
    )
)

Nuisance_parameters = Parameters(
    [
        Parameter(name="db" + str(i), value=0, frozen=False)
        for i in range(ndim_3D_nui)
    ]
)

l_corr = 0.1
# divide sysampliude by the number of spatial bins it was summed over to compute
sysamplitude = np.array(sysamplitude)

bg = (
    dataset_standard.background
    .data.sum(axis=2)
    .sum(axis=1)
)[nuisance_mask_helper]
print(bg)
for ii in np.arange(len(sysamplitude)):
    if sysamplitude[ii] > 0.0:
        sys_percent = sysamplitude[ii]
        
geom_down = nuisance_mask.geom
helper_map = Map.from_geom(geom_down).slice_by_idx(dict(energy=slice(0, 1)))
helper_map2 = helper_map.copy()


def compute_K_matrix(l_deg):
    corr_matrix_spatial = np.identity(ndim_spatial_nui)
    for b_0 in range(ndim_spatial_nui_1D):
        for l_0 in range(ndim_spatial_nui_1D):
            i = b_0 * ndim_spatial_nui_1D + l_0
            C = SkyCoord(
                helper_map.geom.pix_to_coord((l_0, b_0, 0))[0],
                helper_map.geom.pix_to_coord((l_0, b_0, 0))[1],
                frame=geom_down.frame,
            )
            helper_map.data[0, :, :] = C.separation(
                geom_down.to_image().get_coord().skycoord
            ).value
            helper_map2.data = np.zeros(ndim_spatial_nui_1D ** 2).reshape(
                helper_map2.geom.data_shape
            )
            helper_map2.data[0, :, :] = np.exp(
                -0.5 * helper_map.data[0, :, :] ** 2 / l_deg ** 2
            )
            corr_matrix_spatial[i, :] = helper_map2.data.flatten()

    corr_matrix_spectral = np.identity(ndim_spectral_nui)
    for e in range((ndim_spectral_nui)):
        corr_matrix_spectral[e, e] = sysamplitude[nuisance_mask_helper][e] ** 2
    return np.kron(corr_matrix_spectral, corr_matrix_spatial)

correlation_matrix = compute_K_matrix(l_corr)

dataset_N = MapDatasetNuisance(
    background=dataset_standard.background,
    exposure=dataset_standard.exposure,
    psf=dataset_standard.psf,
    edisp=dataset_standard.edisp,
    mask_fit=dataset_standard.mask_fit,
    mask_safe=dataset_standard.mask_safe,
    counts=dataset_standard.counts,
    inv_corr_matrix=np.linalg.inv(correlation_matrix),
    N_parameters=Nuisance_parameters,
    nuisance_mask=nuisance_mask,
)

bkg_model = FoVBackgroundModel(dataset_name=dataset_N.name)
bkg_model.parameters["tilt"].frozen = False

initial_model_hess_N = initial_model_hess.copy()
initial_model_hess_N.append(bkg_model)
dataset_N.models = initial_model_hess_N
print(dataset_N)

fit_N = Fit(store_trace=False)
result_N = fit_N.run([dataset_N])
print(result_N)


import yaml
# writing the fitted MapDatasetNuisance ...
outputfile = '/home/vault/caph/mppi062h/repositories/syserror_3d_bkgmodel/2-source_dataset/GC_0.19'
dataset_N.write(outputfile+f'/{name}_nui_dataset.fits', overwrite = True)
with open(outputfile+f'/{name}nui_model.yml', 'w+') as outfile:
        yaml.dump(dataset_N.models.to_dict(), outfile, default_flow_style=False)
with open(outputfile+f'/{name}nui_bgmodel.yml', 'w+') as outfile:
        yaml.dump(dataset_N.background_model.to_dict(), outfile, default_flow_style=False)
with open(outputfile+f'/{name}nui_par.yml', 'w+') as outfile:
        yaml.dump(dataset_N.N_parameters.to_dict(), outfile, default_flow_style=False)


