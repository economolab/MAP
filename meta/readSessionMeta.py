# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
from pynwb import NWBHDF5IO
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
from time import sleep
import pandas as pd
from scipy import stats
from tqdm import tqdm

from loadDataUtils import *
from plotUtils import *

np.random.seed(123)

%matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## WHAT INFO DO I WANT

# performance
# nTrials (right and left)
# brain regions (ALM + contra Medulla)
# cluster per region ()

path = '/Users/munib/Documents/Economo-Lab/code/map-ephys/one-session'
fn = 'allSessionsMetaData.csv'
df = pd.read_csv(os.path.join(path,fn))



# %%

nSessions = len(df)

# plot number of sessions per brain region

# get all possible regions

def getRegionsDict(df,nSessions):
    regions = dict()
    for isess in range(nSessions):
        regions[isess] = []
        temp = df.regions.iloc[isess]
        tempsplit = temp.split("'")
        for i in range(len(tempsplit)):
            if 'right' in tempsplit[i] or 'left' in tempsplit[i]:
                regions[isess].append(tempsplit[i])
    return regions

regions = getRegionsDict(df,nSessions) # regions per sessions

regionsList = np.unique(sum(regions.values(),[])) # unique regions across all sessions

# %%

# count how many sessions have simulataneous alm and medulla

use = []
for isess in range(nSessions):
    r = regions[isess]
    alm = [s for s in r if "ALM" in s]
    medulla = [s for s in r if "Medulla" in s]
    if not not alm and not not medulla and df.Perf.iloc[isess] >= 70:
        use.append(isess)


subs = (df['sub'].iloc[use])
dates = (df['date'].iloc[use])
nTrials =  (df['nTrials'].iloc[use])
nRightTrials = (df['nRightTrials'].iloc[use])
nLeftTrials = (df['nLeftTrials'].iloc[use])
perf = (df['Perf'].iloc[use])
probeID = (df['probeID'].iloc[use])
probeType = (df['probeType'].iloc[use])
regions_new = (df['regions'].iloc[use])
nUnits = (df['nUnits'].iloc[use])


d = {'sub' : subs,
     'date': dates,
     'nTrials' : nTrials,
     'nRightTrials' : nLeftTrials,
     'nLeftTrials' : nLeftTrials,
    'Perf':perf,
    'probeID':probeID,
    'probeType':probeType,
    'regions':regions_new,
    'nUnits':nUnits}
            

df_new = pd.DataFrame(columns = d)
df_new['sub'] = subs
df_new['date'] = dates
df_new['nTrials'] = nTrials
df_new['nRightTrials'] = nRightTrials
df_new['nLeftTrials'] = nLeftTrials
df_new['Perf'] = perf
df_new['probeID'] = probeID
df_new['probeType'] = probeType
df_new['regions'] = regions
df_new['nUnits'] = nUnits

# df_new.to_csv('ALM_IRN_sessionMeta.csv', encoding='utf-8', index=False)


# %%
# count how many times each region was recorded from

allRegions = sum(regions.values(),[])
regionsCount = []
for i in range(len(regionsList)):
    r = regionsList[i]
    regionsCount.append(allRegions.count(r))
regionsCount = np.array(regionsCount)

regionsList = regionsList[0:int(len(regionsList)/2)]
for i in range(len(regionsList)):
    regionsList[i] = regionsList[i].split()[1]

rcount = regionsCount[0:int(len(regionsCount)/2)]

rcount += regionsCount[int(len(regionsCount)/2)::]

# %% PLOT

x = np.arange(0,len(rcount)) # x-coordinates of your bars
w = 0.8
fig, ax = plt.subplots(figsize=(6,5))
ax.bar(x,
    height=rcount,
    width=w,    # bar width
    tick_label=regionsList,
    # color='g',  # face color transparent
    # edgecolor='w'
    )
plt.xticks(rotation=30)
plt.ylabel('# of recordings')
plt.subplots_adjust(left=0.25,bottom=0.15)
plt.show()

# %%
