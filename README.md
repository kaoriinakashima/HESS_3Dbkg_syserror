Explanation about the structure:

general_config:
    The same configuration will be use for all the analysis and it is stored all the main variables there.
    Please, change the path of the folders according to your needs

0- a set of notebook and scripts to reproduce the plots shown on confluence page


1- this is where the error level calculation is done


2- this is where we apply the error in a dataset for a given source



fixed_material: folder for the basic level estimation result and also for the gamma ray sources to be masked in the analysis



some important comments:
these scripts are considering gammapy-0.19
it is last updated in March 22nd 2022. 
Mainly using the configurations from general_config, only hess1, and hess2, using the bkg model according to BCD scheme, for std_zeta_fullEnclosure configuration, 