import gammapy 
print(f'loaded gammapy version: {gammapy.__version__} ' )
print(f'Supposed to be 0.20 (18-10-2022)' )

#get_ipython().system('jupyter nbconvert --to script 5a-Tutorial_Nui_Par_Fitting_Crab.ipynb')
import pyximport

pyximport.install()
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('pdf')
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
from regions import CircleSkyRegion, RectangleSkyRegion
import yaml
import sys

#sys.path.append(
#    "/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/syserror_3d_bkgmodel/4-Fitting_nuisance_and_model_parameters"
#)
from my_dataset_maps_20 import MapDatasetNuisance
from  my_fit_20 import Fit

source = 'GC_dataset_zeta_5_muoneffTrue_edispTrue'
free_parameters = 'centralspectrumfree' # modelfrozen
print(source, free_parameters)
path = '/home/vault/caph/mppi062h/repositories/HESS_3Dbkg_syserror/2-error_in_dataset'
local_path = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/HESS_3Dbkg_syserror/2-error_in_dataset'


if source == "Crab":

    dataset_standard = MapDataset.read(f'{path}/{source}/stacked.fits')
    dataset_standard = dataset_standard.downsample(4)

    models = Models.read(f"{source}/standard_model.yml")
    
    with open(f"{source}/nui_bgmodel.yml", "r") as ymlfile:
        best_fit_bgmodel = yaml.load(ymlfile, Loader=yaml.FullLoader)
    bkg_model = FoVBackgroundModel(dataset_name=dataset_standard.name)
    bkg_model.parameters['norm'].value = best_fit_bgmodel['spectral']['parameters'][0]['value']
    bkg_model.parameters['tilt'].value = best_fit_bgmodel['spectral']['parameters'][1]['value']
    bkg_model.parameters['norm'].error = best_fit_bgmodel['spectral']['parameters'][0]['error']
    bkg_model.parameters['tilt'].error = best_fit_bgmodel['spectral']['parameters'][1]['error']
    
    models.parameters['lon_0'].frozen = True
    models.parameters['lat_0'].frozen = True
    
    models.append(bkg_model)
    dataset_standard.models = models
    ebins_display = 6,9
    print(dataset_standard)

    
if "GC" in source :
    if source =='GC':
        path_GC ='/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/syserror_3d_bkgmodel/2-source_dataset/GC_0.19'
        dataset_standard = MapDataset.read(f'{path_GC}/20220511_dataset002_hess1_muonflagTrue.fits')
        dataset_standard.stack(MapDataset.read(f'{path_GC}/20220511_dataset002_hess2_muonflagTrue.fits'))
        dataset_standard = dataset_standard.downsample(4)
        
        
        # Define source model for stacked fit
        from gammapy.modeling.models import (PointSpatialModel, 
                                             ExpCutoffPowerLawSpectralModel,
                                             PowerLawSpectralModel,
                                             GaussianSpatialModel,TemplateSpatialModel,
                                            PowerLawNormSpectralModel)
        spatial_model_center = PointSpatialModel(lon_0=359.9439608*u.deg, lat_0=-0.0418969*u.deg, frame='galactic') 
        spectral_model_center = ExpCutoffPowerLawSpectralModel(index=2.14, amplitude="2.55e-12 cm-2 s-1 TeV-1", reference="1 TeV", lambda_='0.093 TeV-1')
        source_model_center = SkyModel(spatial_model=spatial_model_center, spectral_model=spectral_model_center, name="src")

        spatial_model_g09 = PointSpatialModel(lon_0=0.8717549*u.deg, lat_0=0.0767488*u.deg, frame='galactic')
        spectral_model_g09 = PowerLawSpectralModel(index=2.4, amplitude="0.838e-12 cm-2 s-1 TeV-1", reference="1 TeV") # was 3e-12
        source_model_g09 = SkyModel(spatial_model=spatial_model_g09, spectral_model=spectral_model_g09, name="g09")

        spatial_model_1745 = GaussianSpatialModel(lon_0=358.6435538*u.deg, lat_0=-0.5617789*u.deg, sigma=0.179*u.deg, frame='galactic')
        spectral_model_1745 = PowerLawSpectralModel(index=2.57, amplitude="1.73e-12 cm-2 s-1 TeV-1", reference="1 TeV") # 
        source_model_1745 = SkyModel(spatial_model=spatial_model_1745, spectral_model=spectral_model_1745, name="1745")

        spatial_model_1746 = PointSpatialModel(lon_0=0.1384563*u.deg, lat_0=-0.1112664*u.deg, frame='galactic')
        spectral_model_1746 = PowerLawSpectralModel(index=2.17, amplitude="0.18e-12 cm-2 s-1 TeV-1", reference="1 TeV") # was 3e-12
        source_model_1746 = SkyModel(spatial_model=spatial_model_1746, spectral_model=spectral_model_1746, name="1746")

        spatial_model_1746308 = GaussianSpatialModel(lon_0=358.4479799*u.deg, lat_0=-1.1140008*u.deg, sigma=0.162*u.deg, frame='galactic')
        spectral_model_1746308 = PowerLawSpectralModel(index=3.27, amplitude="0.70e-12 cm-2 s-1 TeV-1", reference="1 TeV") #
        source_model_1746308 = SkyModel(spatial_model=spatial_model_1746308, spectral_model=spectral_model_1746308, name="1746308")

        spatial_model_1741 = PointSpatialModel(lon_0=358.2753545*u.deg, lat_0=0.0515537*u.deg, frame='galactic')
        spectral_model_1741 = PowerLawSpectralModel(index=2.30, amplitude="0.21e-12 cm-2 s-1 TeV-1", reference="1 TeV") # was 3e-12
        source_model_1741 = SkyModel(spatial_model=spatial_model_1741, spectral_model=spectral_model_1741, name="1741")

        diffuse_filename = "/home/saturn/caph/mppi043h/diffusiontemplate/cont_pcut_v3.fits"
        diffuse_gal = Map.read(diffuse_filename)
        print(diffuse_gal.geom.axes[0].name)
        template_diffuse = TemplateSpatialModel(
                diffuse_gal, normalize=False,
            filename = diffuse_filename
            )
        diffuse_model = SkyModel(
            spectral_model=PowerLawNormSpectralModel(),
            spatial_model=template_diffuse,
            name="Diffuse Emission",)

        bkg_model = FoVBackgroundModel(dataset_name=dataset_standard.name)
        bkg_model.parameters['tilt'].frozen  = False

        models = Models([source_model_center, source_model_g09, source_model_1745, source_model_1746, 
                         source_model_1746308, source_model_1741, diffuse_model, bkg_model ])

    
    if 'muoneffTrue_edispTrue' in source:
        
        path_GC = '/home/vault/caph/mppi062h/repositories/GC/HESS/datasets_fits'
        dataset_standard = MapDataset.read(f'{path_GC}/{source[3:]}.fits')
        if free_parameters == 'centralspectrumfree':
            models = Models.read(f'{source}/HESS_fixed_models_centralspectrumfree.yml')
        if free_parameters == 'modelfrozen':
            models = Models.read(f'{source}/HESS_fixed_models.yml')
            
        print(models)
        dataset_standard.models = models

    print("loaded the models")
        
    
    j1745_coord = SkyCoord(358.6435538, -0.5617789, unit='deg',frame='galactic')

    skyregion_1745 = RectangleSkyRegion(center=j1745_coord, width=1*u.deg,height=1*u.deg)
    geom_2d = dataset_standard.geoms['geom'].drop('energy')
    dataset_standard.mask_safe.data[:] &= Map.from_geom(geom_2d, 
                                                        data=geom_2d.region_mask([skyregion_1745], 
                                                                                          inside=False).data).data
    dataset_standard.models = models
    ebins_display = 4, 14
    print("masked the dataset")


dataset_standard.counts.sum_over_axes().plot(add_cbar=1)
binsize = dataset_standard.geoms["geom"].width[1] / dataset_standard.geoms["geom"].data_shape[1]
print(
    "spatial binsize = ",
    binsize
)


fit_standarad = Fit(store_trace=False)
try:
    result_standarad = fit_standarad.run([dataset_standard])
except:
    print("no free model parameters")
    
    
res_standard = (
    dataset_standard.residuals("diff/sqrt(model)")
    .slice_by_idx(dict(energy=slice(ebins_display[0],ebins_display[1] )))
    .smooth(0.1 * u.deg)
)
vmax = np.nanmax(np.abs(res_standard.data))
res_standard.plot_grid(add_cbar=1, vmax=vmax, vmin=-vmax, cmap="coolwarm");
kwargs_spectral = dict()
kwargs_spectral["region"] = CircleSkyRegion(
    dataset_standard.geoms["geom"].center_skydir, radius=3 * u.deg
)
kwargs_spectral["method"] = "diff/sqrt(model)"
dataset_standard.plot_residuals(kwargs_spectral=kwargs_spectral);







bg = (
    dataset_standard.background
    .data.sum(axis=2)
    .sum(axis=1)
)

#sysamplitude_std = np.loadtxt(f'{source}/sysamplitude.txt')
# Convert to %:
#sysamplitude_std /= np.sqrt(bg)


sysamplitude_percentage = np.loadtxt((f'{local_path}/{source}/sysamplitude_percentage.txt'))
# Convert to %:
sysamplitude_percentage /= 100
print("sysamplitude_percentage:",sysamplitude_percentage)


#choose between the following:
sigma = sysamplitude_percentage 
#sigma = sysamplitude_std 


emask = sigma >0
print("Estimated systematic uncertainty: ")
print()
print("Ebin               [Counts]    [% BG]")

for i, e in enumerate(dataset_standard.geoms['geom'].axes[0].center.value):
    e_start, e_end = (
        dataset_standard.geoms["geom"].axes[0].edges[ i],
        dataset_standard.geoms["geom"].axes[0].edges[i + 1],
    )
    sys_percent = sigma[i]
    sys_counts =  sigma[i] * bg[i]
    

    print(
        f"{np.round(e_start.value,1):<4} : {np.round(e_end.value,1):<6} TeV:  {np.round(sys_counts,0):<10}  {np.round(sys_percent,3):<5}  "
        )
    
angular_size_file = f'{source}/angular_size.txt'
angular_size = np.loadtxt(angular_size_file)
ndim_spatial = dataset_standard.geoms['geom'].data_shape[1]
print("Current Binning:", ndim_spatial)
possible_downsampling_factors = []
possible_binsizes = []
for i in range(1, ndim_spatial):
    if (ndim_spatial%i == 0):
        possible_downsampling_factors.append(i)
        possible_binsizes.append(binsize[0].value * i)


print(f"Possible downsampling factors: {possible_downsampling_factors}")
print(f"Resulting Binsize: {possible_binsizes}")

downsampling_factor_index = -1
while (possible_binsizes[downsampling_factor_index] > angular_size):
    downsampling_factor_index -=1
downsampling_factor =    possible_downsampling_factors[downsampling_factor_index]

# ##############################
#downsampling_factor = 10
# ##############################


print()
print(f"Chosen Downsampling Factor: \n {downsampling_factor}.")
print(f"This will result in a Binsize of the Nuisance Parameters of \n {possible_binsizes[downsampling_factor_index]} deg.")
print(f"Which is smaller than the observed angular size of the systematics of \n {angular_size} deg.")

geom_down = dataset_standard.downsample(downsampling_factor).geoms['geom']



i_start, i_end = 2,24
nuisance_mask_hand = (
    dataset_standard.geoms["geom"]
    .energy_mask(
        energy_min=dataset_standard.geoms["geom"].axes[0].edges[i_start],
        energy_max=dataset_standard.geoms["geom"].axes[0].edges[i_end],
    )
    .downsample(downsampling_factor)
)


print('Creating Mask for Nuisance Parameters where sysamplitude==0')
nui_mask = np.abs(sigma)>0
print('nui_mask:',nui_mask)
nuisance_mask = Map.from_geom(geom_down, dtype=bool)
for e, n in enumerate(nui_mask):
    nuisance_mask.data[e,:,:] = n
nuisance_mask &= nuisance_mask_hand
emask = nuisance_mask.data.mean(axis=2).mean(axis=1)
emask = list(map(bool,emask))
print(emask)



threshold = 1#10000
bg_map_eaxis = dataset_standard.background.data.sum(axis = 2).sum(axis=1)


print("                                     sys * thresh < stat:")
for i in range(len(bg_map_eaxis)):
    stat =np.round(np.sqrt(bg_map_eaxis[i]) )
    sys = np.round(np.abs(sigma[i]*bg_map_eaxis[i]))
    print(f"BG: {np.round(bg_map_eaxis[i]):<10} pm {stat:<14}   |  {sys}")
    print(f" { ((sys * threshold) < stat):>60}")
    
    
for ii in np.arange(len(sigma)):
    print( np.abs(sigma[ii]) >0.0 , emask[ii])
    
    
# new way to compute correlation matrix to make it symmetric/ invertible
# into Gaussian: sigma **2  * exp(...), where sigma  is the systematic amplitude in %
# if it is saved in terms of std it has to be transformed! 
ndim_spectral_nui = int(sum(emask))
ndim_spatial_nui_1D = geom_down.data_shape[1]
ndim_spatial_nui = ndim_spatial_nui_1D **2
ndim_3D_nui = ndim_spectral_nui *  ndim_spatial_nui

sys_map = Map.from_geom(geom_down).slice_by_idx(dict(energy=slice(0,int(ndim_spectral_nui))))
e = 0
sys_map.data = np.ones_like(sys_map.data)
for ii in np.arange(len(sigma)):
    if np.abs(sigma[ii]) >0.0 and emask[ii]:
        if dataset_standard.npred_background().downsample(downsampling_factor).data[ii,:,:].sum() > 0:
            sys_map.data[e,:,:] *= sigma[ii] **2
            print(sigma[ii])
        e+=1

# Freeze Nuisance parameters at the edges of the analysis
threshold = 1
bg_map  = dataset_standard.background.downsample(downsampling_factor)
bg = bg_map.data[emask].flatten()
stat_err_ = np.sqrt(bg)
Nuisance_parameters = [Parameter(name = "db"+str(i), value =0,frozen = False)  
            if sys_map.data.flatten()[i]  * threshold < stat_err_[i] 
            else  Parameter(name = "db"+str(i), value = 0,frozen = True)
      for i in range(ndim_3D_nui)]
ii = 0
for i in Nuisance_parameters:
    if i.frozen == False:
        ii += 1
print(ii, ' free Nuisance Parameters out of ', ndim_3D_nui)
Nuisance_parameters = Parameters( Nuisance_parameters)
l_corr = 0.08

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
        corr_matrix_spectral[e, e] = sigma[emask][e] ** 2
    return np.kron(corr_matrix_spectral, corr_matrix_spatial)
correlation_matrix = compute_K_matrix(l_corr)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlation_matrix)  # interpolation='nearest')
fig.colorbar(cax);
print("Maximal expected sys amplitude in % of bg:", np.sqrt(correlation_matrix.max() ) * 100)
print("Maximal sigma:", sigma[emask].max()* 100)

name = f'plots/Example_corr_matrix.png'
fig.savefig(name, dpi=300, bbox_inches = 'tight')

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
models_N = models.copy()
models_N.append(bkg_model)
dataset_N.models = models_N
dataset_N.npred_background()
print(dataset_N)
## Check the mask:
Nuisance_parameters_check = Nuisance_parameters.copy()
for N in Nuisance_parameters_check.free_parameters:
    N.value = 1
dataset_N.N_parameters = Nuisance_parameters_check
fig = plt.figure()
dataset_N.N_map().plot_grid(add_cbar =1, vmin = 0, vmax = 1);

dataset_N.N_parameters = Nuisance_parameters

fig = plt.gcf()

name = f'{source}/plots/nuisance_mask.png'
fig.savefig(name, dpi=300, bbox_inches = 'tight')
fig = plt.figure()
print("Fittin nui pars")
fit_N = Fit(store_trace=False)
result_N = fit_N.run([dataset_N])

vmax = np.max(np.abs(dataset_N.N_map().data))
dataset_N.N_map().plot_grid(
    add_cbar=1, vmax=vmax, vmin=-vmax
);

fig = plt.gcf()
name = f'{source}/plots/best_fit_nuis.png'
fig.savefig(name, dpi=300, bbox_inches = 'tight')
fig = plt.figure()
res_standard = (
    dataset_standard.residuals("diff/sqrt(model)")
    .slice_by_idx(dict(energy=slice(i_start, i_end)))
    .smooth(0.1 * u.deg)
)
vmax__ = np.nanmax(np.abs(res_standard.data))
res_standard.plot(add_cbar=1, vmax=vmax__, vmin=-vmax__, cmap="coolwarm");

fig = plt.gcf()
name = f'{source}/plots/res_standard.png'
fig.savefig(name, dpi=300, bbox_inches = 'tight')
fig = plt.figure()
res_N = (
    dataset_N.residuals("diff/sqrt(model)")
    .slice_by_idx(dict(energy=slice(i_start, i_end)))
    .smooth(0.1 * u.deg)
)
res_N.plot(add_cbar=1, vmax=vmax__, vmin=-vmax__, cmap="coolwarm");

fig = plt.gcf()
name = f'{source}/plots/res_N.png'
fig.savefig(name, dpi=300, bbox_inches = 'tight')
fig = plt.figure()

res_N = (
    dataset_N.residuals("diff/sqrt(model)")
    .slice_by_idx(dict(energy=slice(ebins_display[0],ebins_display[1] )))
    .smooth(0.1 * u.deg)
)
vmax = np.nanmax(np.abs(res_standard.data))
res_N.plot_grid(add_cbar=1, vmax=vmax, vmin=-vmax, cmap="coolwarm");


res_standard = (
    dataset_standard.residuals("diff/sqrt(model)")
    .slice_by_idx(dict(energy=slice(i_start, i_end)))
    .data.flatten()
)
res_N = (
    dataset_N.residuals("diff/sqrt(model)")
    .slice_by_idx(dict(energy=slice(i_start, i_end)))
    .data.flatten()
)

_, bins, _ = plt.hist(
    res_standard,
    bins=50,
    alpha=0.4,
    label="Standard: \n$\mu$ = {:.3} \n$\sigma$ = {:.3}".format(
        np.nanmean(res_standard), np.nanstd(res_standard)
    ),
)
plt.hist(
    res_N,
    bins=bins,
    alpha=0.4,
    label="Nuisance: \n$\mu$ = {:.3} \n$\sigma$ = {:.3}".format(
        np.nanmean(res_N), np.nanstd(res_N)
    ),
)
plt.yscale("log")
plt.legend()
plt.xlabel("Significance")
plt.ylabel("Amount");


ax = dataset_standard.plot_residuals_spectral(method = kwargs_spectral["method"],
                                    region = kwargs_spectral["region"],
                                        color = 'red',
                                        label = "Standard Analysis")
ylim = ax.get_ylim()
dataset_N.plot_residuals_spectral(ax = ax ,method = kwargs_spectral["method"],
                                    region = kwargs_spectral["region"],
                                         color = 'green',
                                         label = "With Nuisance Par.")
ax.set_ylim(ylim[0], ylim[1])
eax = dataset_standard.geoms['geom'].axes[0].edges.value
print(eax)
ax.set_xlim(eax[6], eax[-1])
ax.legend(loc = 'lower right')
fig = plt.gcf()
name = f'plots/2_Spectral_Residuals.png'
fig.savefig(name, dpi=300, bbox_inches = 'tight')
name = f'plots/2_Spectral_Residuals.pdf'
fig.savefig(name, dpi=300, bbox_inches = 'tight')


dataset_pseudo = dataset_standard.copy()
dataset_pseudo.background = dataset_N.npred_background().copy()
bkg_model = FoVBackgroundModel(dataset_name=dataset_pseudo.name)
bkg_model.parameters["norm"].frozen = True
models_pseudo = models.copy()
models_pseudo.append(bkg_model)
dataset_pseudo.models = models_pseudo

fit_pseudo = Fit(store_trace=False)
try:
    result_pseudo = fit_pseudo.run([dataset_pseudo])
except:
    print("no free parameters")
print(dataset_pseudo)


print(" with nuisance")
print("(without nuisance)")

for p_N, p_stand , p_pseudo in zip(dataset_N.models.parameters,
                                   dataset_standard.models.parameters,
                                  dataset_pseudo.models.parameters):
    print()
    print('='*50)
    print(p_N.name , p_N.frozen)
    print('-'*50)
    print(" {:.4} +- {:.3} +- {:.3}".format(p_N.value, float(p_pseudo.error) ,
                                            float(p_N.error)- float(p_pseudo.error)  )   ) 
    print('({:.4} +- {:.3})'.format(p_stand.value, float(p_stand.error) ))


added =  '00'+str(int(binsize[0].value* 100)) + '_' + str(i_start) + str(i_end)
print(added)

import yaml
save = 1
path_local_repo = '/home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/syserror_3d_bkgmodel/2-source_dataset'


if save:
    print(f"save in: {path_local_repo}/{source}/nui_dataset_{added}{free_parameters}.fits" )
    print(f"and: {path_local_repo}/nui_bgmodel_{added}{free_parameters}.yml ")


    # save for now in this folder
    dataset_N.write(f'{path_local_repo}/{source}/nui_dataset_{added}{free_parameters}.fits', overwrite = True)
    with open(f'{path_local_repo}/{source}/nui_par_{added}{free_parameters}.yml', 'w') as outfile:
            yaml.dump(dataset_N.N_parameters.to_dict(), outfile, default_flow_style=False,
                     )
    with open(f'{path_local_repo}/{source}/nui_bgmodel_{added}{free_parameters}.yml', 'w') as outfile:
            yaml.dump(dataset_N.background_model.to_dict(), outfile, default_flow_style=False,
                     )
    with open(f'{path_local_repo}/{source}/nui_model_{added}{free_parameters}.yml', 'w') as outfile:
            yaml.dump(dataset_N.models.to_dict(), outfile, default_flow_style=False,
                     )        

    with open(f'{path_local_repo}/{source}/pseudo_bgmodel_{added}{free_parameters}.yml', 'w') as outfile:
            yaml.dump(dataset_pseudo.background_model.to_dict(), outfile, default_flow_style=True)
    with open(f'{path_local_repo}/{source}/pseudo_model_{added}{free_parameters}.yml', 'w') as outfile:
            yaml.dump(dataset_pseudo.models.to_dict(), outfile, default_flow_style=True)    
