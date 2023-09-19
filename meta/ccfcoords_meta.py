# READ CCF COORDS CSVs and save meta data about regions in a separate csv
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import defaultParams
import utils

import sys
import importlib
from tqdm import tqdm
from scipy import stats
import pandas as pd
import nrrd

from time import sleep
import os
from pynwb import NWBHDF5IO
import numpy as np

from ipywidgets import widgets
import matplotlib as mpl
mpl.rcParams['font.size'] = 12
import matplotlib.pyplot as plt
import seaborn as sns
import opinionated as op # https://github.com/MNoichl/opinionated 
# plt.style.use('dark_background')
# plt.style.use(
#     'https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
%matplotlib widget



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def loadCoordinates(dataDir,sub,date):
    sessionList = os.listdir(os.path.join(dataDir,"sub-"+sub))
    sessions = [s for s in sessionList if date in s]
    sessions = [s for s in sessions if 'csv' in s] # keep ccfcoords.csv from list
    coordsFile = sessions[0] if len(sessions) == 1 else sessions 

    df = pd.read_csv(os.path.join(dataDir,'sub-'+sub,coordsFile))
    
    return df

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PARAMETERS
# 'par' is associated with loading and preprocessing the data
# 'params' contains properties of the data itself (trials, units ids, etc.)

if os.name == 'nt':  # Windows PC (office)
    dataDir = r'C:\Users\munib\Documents\Economo-Lab\data'
else:  # Macbook Pro M2
    # dataDir = '/Volumes/MUNIB_SSD/Economo-Lab/data/'
    dataDir = '/Users/munib/Economo-Lab/data'

proj = "map" # subdirectory of dataDir

path = '/Users/munib/Documents/Economo-Lab/code/map/meta'
fn = 'ALM_IRN_sessionMeta.csv'
sessdf = pd.read_csv(os.path.join(path,fn))

subs = [str(sub) for sub in sessdf['sub']]
dates = [str(date) for date in sessdf['date']]

par = defaultParams.getDefaultParams() 

par.behav_only = 1

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LOAD DATA

acronyms = ['MOs','IRN','MOp','MY','XII','VII','GRN','MARN','PARN','LRN']
nSessions = len(subs)
nUnits_df = pd.DataFrame(np.zeros((nSessions,len(acronyms))),columns=acronyms)
for i in range(nSessions):
    sub = subs[i]
    date = dates[i]

    df = loadCoordinates(os.path.join(dataDir,proj),sub,date)
    
    for j,a in enumerate(acronyms):
        nUnits_df.iloc[i,j] = np.sum(df.acronym.str.contains(a)).astype(int)
        
        
s = np.array(subs)
d = np.array(dates)
subdf = pd.DataFrame(np.vstack((s,d)).T,columns=['sub','date'])

savedf = pd.concat((subdf,nUnits_df),axis=1)

# %%
# save
savedir = '/Users/munib/Documents/Economo-Lab/code/map/meta'
fn = 'MOs_MY_nUnits.csv'
savedf.to_csv(os.path.join(savedir,fn),index=False)
print('Saved ' + os.path.join(savedir,fn) + ' !!!')
# %% STACKED BAR PLOT

path = '/Users/munib/Documents/Economo-Lab/code/map/meta'
fn = 'MOs_MY_nUnits.csv'
df = pd.read_csv(os.path.join(path,fn))
# %%

with plt.style.context('opinionated_rc'):
    fig,ax = plt.subplots(figsize=(5,4),constrained_layout=True)
    newdf = df.iloc[:,2:]
    ax = newdf.plot.bar(stacked=True,ax=ax)
    plt.legend(fontsize=8,loc='best')
    ax.set_xlabel('Session',fontsize=12)
    ax.set_ylabel('# Units',fontsize=12)
    # plt.xticks(fon?)



# %%
