
# outdated - this is when i thought nwbfile.units.xpos/ypos were the ccf cooordinates

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import sys
import importlib
from tqdm import tqdm
from scipy import stats
import pandas as pd

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

import sys
sys.path.append(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\basic-analysis')
import defaultParams
import utils


# %%
# LOAD DATA
path = '/Users/munib/Documents/Economo-Lab/code/map-ephys/meta'
fn = 'ALM_IRN_sessionMeta.csv'
df = pd.read_csv(os.path.join(path,fn))
nSessions = df.shape[0]

if os.name == 'nt':  # Windows PC (office)
    dataDir = r'C:\Users\munib\Documents\Economo-Lab\data'
else:  # Macbook Pro M2
    # dataDir = '/Volumes/MUNIB_SSD/Economo-Lab/data/'
    dataDir = '/Users/munib/Economo-Lab/data'

proj = "map-ephys" # subdirectory of dataDir
dataDir = os.path.join(dataDir, proj)

for isess in range(nSessions):
    sub = str(df.iloc[isess]['sub'])
    date = str(df.iloc[isess]['date'])
    
    # find nwb file for given sub and date
    sessions = utils.findNWB(dataDir, sub, date)
    # returns a list if more than one session provided, but this code only works for one session at the moment
    sessions = sessions[0] if len(sessions) == 1 else sessions 
    
    # LOAD DATA
    nwbfile = NWBHDF5IO(os.path.join(dataDir, "sub-"+sub, sessions)).read() # load nwb file

    break
    # UNIT LOCATIONS AND THEIR POSITIONS ON PROBE

    xpos = nwbfile.units['unit_posx'].data[:]
    ypos = nwbfile.units['unit_posy'].data[:]


    egroup = nwbfile.units['electrode_group'].data[:] # one entry per unit
    nUnits = len(egroup)
    electrodes_dict = []
    probe_dict = []
    for i in range(nUnits):
        locstring = egroup[i].location
        electrodes_dict.append( dict(eval(locstring)) )
        electrodes_dict[i]['xpos'] = xpos[i]
        electrodes_dict[i]['ypos'] = ypos[i]
        
        estring = egroup[i].description
        probe_dict.append( dict(eval(estring)) )
        
    electrodes_df = pd.DataFrame.from_records(electrodes_dict)
    probe_df = pd.DataFrame.from_records(probe_dict)

    units_df = pd.concat([electrodes_df, probe_df], axis=1)


    #
    # PLOT
    cols = utils.Colors()
    # cols = np.array([[117,221,221],
    #                 [147,104,183],
    #                 [231,90,124],
    #                 [255,179,15],
    #                 [0,0,0]]) / 255
    with plt.style.context('opinionated_rc'):
        # groups = units_df.groupby('brain_regions')
        groups = units_df.groupby('probe')
        nRows=2 if len(groups) <=4 else 3
        fig,ax = plt.subplots(nRows,2,constrained_layout=True, figsize=(7,5))
        ax = ax.ravel()
        ct = 0
        for name, group in groups:
            reg = np.array(group.brain_regions)[0] 
            if 'alm' in str.lower(reg):
                cc = cols.alm
            elif 'striatum' in str.lower(reg):
                cc = cols.striatum
            elif 'midbrain' in str.lower(reg):
                cc = cols.midbrain
            elif 'medulla' in str.lower(reg):
                cc = cols.medulla
            
            ax[ct].scatter(group.xpos,group.ypos,s=1,color=cc,label=name,alpha=1)
            
            
            t = np.array(group.brain_regions)[0] + ', ' + \
                np.array(group.ap_location)[0] + ' AP,' + \
                np.array(group.ml_location)[0] + ' ML' + \
                '\n' + np.array(group.depth)[0] + ' D, ' + \
                np.array(group.probe_type)[0]
                
            ax[ct].set_title(t,fontsize=9,color=cc)
            ct += 1
        for a in ax:
            a.set_xlim([-30, 900])
            a.tick_params(axis='both', which='major', labelsize=12)
        # plt.scatter(xpos,ypos,s=2)
        # ax.set_xlabel('xpos (um)')
        # ax.set_ylabel('ypos (um)')
        fig.supxlabel('xpos (um)',fontsize=10)
        fig.supylabel('ypos (um)',fontsize=10)
        for iax in range(ct,len(ax)):
            plt.delaxes(ax[iax])
        plt.show()
        
    
    # fpth = os.path.join(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs',sub,date)
    # fname = 'unit-locations-by-probe' + '_' + date
    # utils.mysavefig(fpth,fname)
    # fpth = os.path.join(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs\unitLocations')
    # fname = sub + '_' + date
    # utils.mysavefig(fpth,fname)

# %%