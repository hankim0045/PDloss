#####################################################################################
# PDloss_online zv model (Run 8 Mar 2024; n = 97; samples = 30000)
# Exclude Ps who failed the attention check during BISBAS
# Quiz threshold = 2 per quiz (total 4)
# There were 2 Qs during prac. This thrsh includes Ps who tried at max 
# 2 times (per Q) before getting the Qs correct.
# Run 4 chains, each with 30k samples & burn 10%
# Let HDDM estimate outliers, instead of setting it at p_outlier = 0.05
# v_reg = {'model': 'v ~ 1 + stim + condition', 'link_func': lambda x: x}  
# z_reg = {'model': 'z ~ 1 + condition', 'link_func': lambda x: x}
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
quizthrsh = 4

orgdata = hddm.load_csv(f'/project/ycleong/users/kimhannah/PDloss_online/scripts/3_ddm/PDloss_online_BehaviourData_fs.csv')

orgdata


## Filter out subjects to be excluded from HDDM
# Note: Invalid trials are already marked (i.e., all rows have NaNs) in the csv file

# Create a boolean mask indicating which elements in the subj_idx column are NaN
discardtrial = np.isnan(orgdata['subj_idx'])

# Filter nan rows out
discarded_nan = orgdata[~discardtrial]

# Filter nan rows out
discarded_nan = orgdata[~discardtrial]

# Filter out subjects who tried too many times before getting the quiz correct
discarded_quiz = discarded_nan[discarded_nan['quiz'] <= quizthrsh]

# Filter out subjects who failed the attention check
finaldata = discarded_quiz[discarded_quiz['attnchk'] == 1]


## Specify model
db_filename = (f'PDloss_online_zv_97_8Mar24_{iteration}.db')
model_filename = (f'PDloss_online_zv_97_8Mar24_{iteration}')

v_reg = {'model': 'v ~ 1 + stim + condition', 'link_func': lambda x: x}  
z_reg = {'model': 'z ~ 1 + condition', 'link_func': lambda x: x}
reg_descr = [v_reg, z_reg]

zv_model = hddm.models.HDDMRegressor(finaldata, reg_descr,
                                       bias=True, include='p_outlier',
                                       group_only_regressors=False)

zv_model.find_starting_values()
zv_model.sample(samples, burn=burnnum, thin=2, dbname=db_filename, db='pickle') 
zv_model.save(model_filename)
