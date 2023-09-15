# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from time import sleep
import pandas as pd
from scipy import stats
from tqdm import tqdm

import sys
sys.path.append(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\basic-analysis')

import utils

np.random.seed(123)

%matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## WHAT INFO DO I WANT

# performance
# nTrials
# nProbes
# brain regions
# cluster per region

df = pd.DataFrame()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## PARAMETERS
allparams = dict() # parameters for all sessions

allparams['alignEvent'] = 'go_start_times'

# units
allparams['lowFR']      = 1; # remove clusters with a mean firing rates across all trials less than this value
allparams['quality']    = ['good','multi']; # unit qualities to use

# set conditions to calculate PSTHs for
hit          = 'outcome == "hit"'
miss         = 'outcome == "miss"'
no           = 'outcome == "ignore"'
R            = 'trial_instruction == "right"'
L            = 'trial_instruction == "left"'
no_early     = 'early_lick == "no early"'
no_autowater = 'auto_water == 0'
no_freewater = 'free_water == 0'
no_stim      = 'photostim_duration == "N/A"' 
stim         = 'photostim_duration != "N/A"' 

allparams['condition'] = []
# (0) not early, no auto/free water, no stim
allparams['condition'].append(no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
# (1) right, not early, no auto/free water, no stim
allparams['condition'].append(R + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
# (2) left, not early, no auto/free water, no stim
allparams['condition'].append(L + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## SET PARAMETERS TO LOAD ONE SESSION
if os.name == 'nt': # Windows PC (office)
    dataDir = r'C:\Users\munib\Documents\Economo-Lab\data'
else: # Macbook Pro M2
    dataDir = '/Users/munib/Economo-Lab/data'
    
proj = "map-ephys"
dataDir = os.path.join(dataDir,proj)

subList = os.listdir(dataDir)

for isub,sub in (enumerate(subList)):
    subDir = os.path.join(dataDir,subList[isub])
    if os.path.isdir(subDir):
        
        # sub = sub
        sessions = os.listdir(subDir)
        
        for isess,sess in (enumerate(sessions)):
            # LOAD DATA
            nwbfile = NWBHDF5IO(os.path.join(dataDir,sub,sess)).read()
            trials_df = nwbfile.trials.to_dataframe()
            units_df = nwbfile.units.to_dataframe()
            units_df = utils.addElectrodesInfoToUnits(units_df)
            fs = stats.mode((nwbfile.units.sampling_rate.data[:]),keepdims=False)[0]
            ## FIND AND SUBSET SESSION-SPECIFIC DATA
            params = dict() # session specific parameters 
            ## FIND TRIALS FOR EACH CONDITION
            params['nTrials'] = trials_df.shape[0]
            params['trials'] = utils.findTrials(nwbfile,trials_df,allparams['condition'])
            # [all trials, not early, not early right, not early left]
            nTrials = [params['nTrials'],params['trials'][0].shape[0],params['trials'][1].shape[0],params['trials'][2].shape[0]]
            ## FIND CLUSTERS
            brain_regions = units_df.brain_regions.unique()
            allparams['regions'] = brain_regions
            units_df, params = utils.findClusters(units_df,allparams,params) # subset units based on lowFR, quality, and region
                    
            ## PERFORMANCE
            cond2use = np.arange(0,3)
            perf = utils.getPerformance(trials_df,params['trials'],cond2use)
            # (all, right, left)
            
            probeid = list(units_df.probe.unique())
            probe_type = [units_df.probe_type[units_df.probe == probeid[p]].iloc[0] for p,_ in enumerate(probeid)]
            
            d = {'sub': list(units_df.subject_id.unique())[0],
                'date': nwbfile.timestamps_reference_time.date().strftime("%Y%m%d"),
                'nTrials': nTrials[0],
                'nRightTrials': nTrials[1],
                'nLeftTrials': nTrials[2],
                'Perf':perf[0],
                'probeID':probeid,
                'probeType':probe_type,
                'regions':brain_regions,
                'nUnits':[params['cluid'][r].shape[0] for r in brain_regions]}
            
            df = df.append(d,ignore_index = True)
            


df.to_csv('allSessionsMetaData_v2.csv', encoding='utf-8', index=False)

