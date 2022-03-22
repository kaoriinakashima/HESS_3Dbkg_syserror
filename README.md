Before you start: in general_config you need to change the path to the repository, and where the FITS files are stored for you. 
The bkg model considering muoneff, is stored in ECAP cluster, so mainly people in ECAP have access to it. However, the error estimation is independent of it. 

If you just want to calculate the error level for a given runlist, go to: folder2, scrip2, that is HESS_3Dbkg_syserror/2-error_in_dataset/2-estimating_error.ipynb and follow the instructions there.

Elif you want to create a dataset with the muoneff model, calculate the systematic error, apply the nuisance parameters, go to HESS_3Dbkg_syserror/2-error_in_dataset/ and run all the scripts there in order. 
Note: in each script you need to add some specific information of your analysis, mainly the source name, what are the gammaray sources in the FoV, what is the source model and etc, it is always in the beginning of the script.

else, if you want to understand the whole process of how the error calculation is done, start with folder 1, and go through all of it. The results are stored in 'fixed material'.

------------------------------------------------------------------
some important comments:
these scripts are considering gammapy-0.19
it is last updated in March 22nd 2022. 
Mainly using the configurations from general_config, only hess1, and hess2, using the bkg model according to BCD scheme, for std_zeta_fullEnclosure configuration.

-------------------------------------------------------------------

Explanation about the structure:

general_config:
    The same configuration will be use for all the analysis and it is stored all the main variables there.
    Please, change the path of the folders according to your needs

0- a set of notebook and scripts to reproduce the plots shown on confluence page


1- this is where the error level calculation is done


2- this is where we apply the error in a dataset for a given source


fixed_material: folder for the basic level estimation result and also for the gamma ray sources to be masked in the analysis



