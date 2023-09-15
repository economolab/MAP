
# %%%
import os
from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import pandas as pd

dataDir = r"C:\Users\munib\Documents\Economo-Lab\data"
proj = r"map-ephys"
sub = r'sub-484677'
date = r'sub-484677_ses-20210420T170516_behavior+ecephys+ogen.nwb'
fpath = os.path.join(dataDir,proj,sub,date)
nwbfile = NWBHDF5IO(fpath).read()

units_df = nwbfile.units.to_dataframe()
nUnits = units_df.shape[0]

# %%%
electrodes_dicts = []
electrode_group_dicts = []
for i in range(nUnits):
    # get data from electrodes for current unit
    estring = estring = units_df.electrodes.iloc[i].location.to_list()[0]
    electrodes_dicts.append( dict(eval(estring)) )
    
    # get data from electrodes for current unit
    estring = units_df.electrode_group.iloc[i].location
    electrode_group_dicts.append( dict(eval(estring)) )
    
# store info in dataframes
electrodes_df = pd.DataFrame.from_records(electrodes_dicts)
electrode_group_df = pd.DataFrame.from_records(electrode_group_dicts)

# print the unique brain regions 
print(np.unique(electrodes_df['brain_regions'])) # prints ['right Striatum']
print(np.unique(electrode_group_df['brain_regions'])) # prints ['left Medulla' 'right ALM' 'right Midbrain' 'right Striatum']

# fraction of units that have same brain_regions value
np.sum( electrodes_df['brain_regions'] == electrode_group_df['brain_regions'] ) / nUnits
# %%
