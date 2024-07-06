#####################################################################################
# PDloss null model (Run on 16 Jan 2024; n = 48; samples = 30000)
# Run 4 chains, each with 30k samples & burn 10%
# Let HDDM estimate outliers, instead of setting it at p_outlier = 0.05
# v_reg = {'model': 'v ~ 1 + stim', 'link_func': lambda x: x}  
# z_reg = {'model': 'z ~ 1', 'link_func': lambda x: x}
#####################################################################################
import sys
iteration = sys.argv[1]
print(f'Iteration = {iteration}')


## Load necessary packages & data
import pandas as pd 
import matplotlib.pyplot as plt 
import hddm 
import numpy as np 
import patsy

samples = 30000
burnnum = samples/10

orgdata = hddm.load_csv(f'/project/ycleong/users/kimhannah/PDloss/scripts/3_ddm/PDloss_BehaviourData_master_fs.csv')

orgdata


## Filter out subjects to be excluded from HDDM
# Note: Invalid trials are already marked (i.e., all rows have NaNs) in the csv file

# Create a boolean mask indicating which elements in the subj_idx column are NaN
discardtrial = np.isnan(orgdata['subj_idx'])

# Filter nan rows out
finaldata = orgdata[~discardtrial]

finaldata


## Specify model
db_filename = (f'PDloss_null_16Jan24_{iteration}.db')
model_filename = (f'PDloss_null_16Jan24_{iteration}')

v_reg = {'model': 'v ~ 1 + stim', 'link_func': lambda x: x}  
z_reg = {'model': 'z ~ 1', 'link_func': lambda x: x}
reg_descr = [v_reg, z_reg]

null_model = hddm.models.HDDMRegressor(finaldata, reg_descr,
                                       bias=True, include='p_outlier',
                                       group_only_regressors=False)

null_model.find_starting_values()
null_model.sample(samples, burn=burnnum, thin=2, dbname=db_filename, db='pickle') 
null_model.save(model_filename)


## Compute DIC (deviance information criterion)
print("Model DIC: %f" % null_model.dic)


## Compute DICc (deviance information criterion, corrected)
DICc = null_model.dic_info['deviance'] + null_model.dic_info['pD'] + null_model.dic_info['pD']
print("Model DICc: %f" % DICc)
