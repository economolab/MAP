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
fn = 'ALM_IRN_sessionMeta.csv'
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

both = 0
right = 0
left = 0
for i in range(nSessions):
    r = regions[i]
    lalm = 'left ALM' in r
    ralm = 'right ALM' in r
    lm = 'left Medulla' in r
    rm = 'right Medulla'  in r
    if lalm and ralm and lm and rm:
        both += 1
    elif lalm and rm:
        left += 1
    elif ralm and lm:
        right += 1
# %%

lab = ['Both', 'ALM_R + Med_L', 'ALM_L + Med_R']
x = np.arange(0,len(lab)) # x-coordinates of your bars
y = [both, right, left]
w = 0.8
fig, ax = plt.subplots(figsize=(3,4))
ax.bar(x,
    height=y,
    width=w,    # bar width
    tick_label=lab,
    color=(0.19607843, 0.65882353, 0.48235294),  # face color transparent
    # edgecolor='k'
    )
plt.xticks(rotation=30)
plt.ylabel('# of sessions')
plt.subplots_adjust(left=0.25,bottom=0.25)
plt.show()

# fpth = r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs'
# fname = 'ALM_IRN_sessions'
# mysavefig(fpth,fname)

# %% number of units per region per session
# stacked bar plot

X = ['left ALM' , 'right ALM' , 'left Medulla' , ' right Medulla']
Y = np.zeros((nSessions,len(X)))

both = 0
right = 0
left = 0
for i in range(nSessions):
    r = regions[i]
    lalm = 'left ALM' in r
    ralm = 'right ALM' in r
    lm = 'left Medulla' in r
    rm = 'right Medulla'  in r
    if lalm:
        Y[i,0] += 1 
    if ralm:
        Y[i,1] += 1 
    if lm:
        Y[i,2] += 1
    if rm:
        Y[i,3] += 1
        

# plt.style.use('dark_background')
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')

fig, ax = plt.subplots(figsize=(5,10))
im = ax.pcolormesh(Y, edgecolors='k', linewidth=2)
ax.set_aspect('equal')

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(0.5,len(X)), labels=X, fontsize=14)
ax.set_yticks(np.arange(Y.shape[0]), labels=np.arange(Y.shape[0]))

plt.setp(ax.get_xticklabels(), rotation=60, ha="right",
         rotation_mode="anchor")


# plt.xlabel(X)
plt.ylabel('Session number', fontsize=15)
fig.tight_layout()
plt.show()


# fpth = r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs'
# fname = 'ALM_IRN_sessions_map'
# mysavefig(fpth,fname,format='svg')
# %%
