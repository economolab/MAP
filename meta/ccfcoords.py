# SAVE CCF COORDS FOR ALL SESSIONS
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
# RERUN THIS WHEN YOU MAKE UPDATES TO ANY OF THESE MODULES AND DON'T WANT TO RESTART KERNEL
_ = importlib.reload(sys.modules['utils'])
_ = importlib.reload(sys.modules['defaultParams'])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### SAVE A FIGURE
# fpth = os.path.join(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs',sub,date)
# fname = plt.gca().get_title() + '_' + date
# plotUtils.mysavefig(fpth,fname)

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

if os.name == 'nt':  # Windows PC (office)
    path = r'/Users/munib/Documents/Economo-Lab/code/map/meta'
else:  # Macbook Pro M2
    # dataDir = '/Volumes/MUNIB_SSD/Economo-Lab/data/'
    path = '/Users/munib//Economo-Lab/code/map/meta'
fn = 'ALM_IRN_sessionMeta.csv'
df = pd.read_csv(os.path.join(path,fn))

subs = [str(sub) for sub in df['sub']]
dates = [str(date) for date in df['date']]

par = defaultParams.getDefaultParams() 

par.behav_only = 1

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LOAD DATA

nSessions = len(subs)
for i in range(nSessions):
    sub = subs[i]
    date = dates[i]

    nwbfile, units_df, trials_df, trialdat, psth, params = \
        utils.loadData(os.path.join(dataDir, proj),sub,date,par,behav_only=par.behav_only)
    # nwbfile - the raw data file in read only mode
    # units_df - dataframe containing info about neurons/units
    # trialdat - dict containing single trial firing rates (trialdat[region] = (time,trials,units))
    # psth - dict of PSTHs (psth[region] = (time,units,conditions))
    # params - session-specific params used for analyses



    # save ccf coords of each unit/electrode to csv
    csvdir = os.path.join(dataDir,proj) # where to save results
    if os.name == 'nt':  # Windows PC (office)
        ccfdir = r'C:\Users\munib\Documents\Economo-Lab\code\map\allenccf'
    else:
        ccfdir = '/Users/munib/Economo-Lab/code/map/allenccf'
        
    coords_df = utils.saveCCFCoordsAndRegion(nwbfile,csvdir,ccfdir,sub,date)


# %%
