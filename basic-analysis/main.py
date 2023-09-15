
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import defaultParams
import utils

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

par = defaultParams.getDefaultParams() 

# change any default params below
# par.regions = ['left ALM',  'left Striatum', 'right Medulla','left Midbrain']
par.regions = ['left ALM']


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LOAD DATA

if os.name == 'nt':  # Windows PC (office)
    dataDir = r'C:\Users\munib\Documents\Economo-Lab\data'
else:  # Macbook Pro M2
    # dataDir = '/Volumes/MUNIB_SSD/Economo-Lab/data/'
    dataDir = '/Users/munib/Economo-Lab/data'

proj = "map-ephys" # subdirectory of dataDir
dataDir = os.path.join(dataDir, proj)

sub = '484676' # subject/animal id
date = '20210420' # session date

behav_only = 1 # 1-trialdat,psth,units_df=NaN, 0-preprocess neural data

nwbfile, units_df, trials_df, trialdat, psth, params = \
    utils.loadData(dataDir,sub,date,par,behav_only=behav_only)
# nwbfile - the raw data file in read only mode
# units_df - dataframe containing info about neurons/units
# trialdat - dict containing single trial firing rates (trialdat[region] = (time,trials,units))
# psth - dict of PSTHs (psth[region] = (time,units,conditions))
# params - session-specific params used for analyses
    
# to get registered coordinates, you have to do
p = nwbfile.units[0]['electrodes'][0] # probe this unit was recorded on
p = []
for i in range(len(nwbfile.units)):
    p.append(int(nwbfile.units[i]['electrodes']))
<<<<<<< HEAD
# np.sum(np.isin(np.unique(p),np.arange(0,len(nwbfile.electrodes))))
=======


>>>>>>> 7d8046c9da50220ee0b1f3ba74a03fcbb8fbc349
# %%
x = []
y = []
z = []
for i in range(len(p)):
<<<<<<< HEAD
    x.append(nwbfile.electrodes[p[i]].x.item())
    y.append(nwbfile.electrodes[p[i]].y.item())
    z.append(nwbfile.electrodes[p[i]].z.item())
    
coords = np.vstack((np.array(z),np.array(y),np.array(x))).T # (nUnits,3)

# np.save('coords'+sub+'_'+date,coords)
    
# x = np.array(x)/1000
# y = np.array(y)/1000
# z = np.array(z)/1000
# # %%
# with plt.style.context('opinionated_rc'):
#     fig = plt.figure(figsize=(4,3))
#     ax = fig.add_subplot(projection='3d')
#     ax.scatter(x, y, z, s=5)
    
#     ax.set_xlabel('AP')
#     ax.set_ylabel('DV')
#     ax.set_zlabel('ML')    
#     ax.set_xlim((0,13)) # ap
#     ax.set_ylim((0,8))  # dv
#     ax.set_zlim((0,11.5)) # ml
#     ax.view_init(elev=133, azim=90)
=======
    x.append( nwbfile.electrodes[p[i]].x.item() )
    y.append( nwbfile.electrodes[p[i]].y.item() )
    z.append( nwbfile.electrodes[p[i]].z.item() )

coords = np.vstack((np.array(z),np.array(y),np.array(x))).T


# np.save('coords'+sub+'_'+date, coords)
>>>>>>> 7d8046c9da50220ee0b1f3ba74a03fcbb8fbc349


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
# TODO: GET KINEMATICS


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PERFORMANCE
cols = utils.Colors()

# cond2use = [0,1,2]
# c = [(0.3,0.3,0.3), cols.rhit, cols.lhit]
# labels = ['all', 'R', 'L']

cond2use = [3,4,7,8]
c = [cols.rhit, cols.lhit, cols.rmiss, cols.lmiss]
labels = ['R', 'L', 'Rin', 'Lin']

perf = utils.taskPerformance(trials_df, params.trialid, cond2use, c, labels, plot=1)

# fpth = os.path.join(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs',sub,date)
# fname = 'performance' + '_inactivation'
# utils.mysavefig(fpth,fname)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# LICK RASTER
_ = importlib.reload(sys.modules['utils'])
cond2plot = [3,4,7,8]
cols = utils.Colors()
c = [cols.rhit, cols.lhit, cols.rmiss, cols.lmiss]
labels = ['R','L','Rin','Lin']
utils.lickRaster(nwbfile, trials_df, par, params, cond2plot, c, labels)

# fpth = os.path.join(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs',sub,date)
# fname = 'lickRaster_hit_inactivation'
# utils.mysavefig(fpth,fname)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# PLOT PSTHs
region = 'left ALM'
# cond2plot = [3,4]
cond2plot = [1,2,7,8]
cols = utils.Colors()
# c = [cols.rmiss, cols.lmiss, cols.rhit, cols.lhit] 
c = [cols.rhit, cols.lhit, cols.rmiss, cols.lmiss] 
lw = [2,2,1,1]
labels = ['r','l','rin','lin']
utils.plotPSTH(trialdat, region, cond2plot, c, lw, 
                   params, par, units_df, nwbfile,
                   legend=labels, plotRaster=1,plotWave=1
                   )

# fpth = os.path.join(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs',sub,date)
# fname = plt.gcf().axes[0].get_title() + '_inactivation'
# utils.mysavefig(fpth,fname)




# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# -%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# EVERYTHING UNDER HERE NEEDS TO BE CLEANED UP AND ADDED TO utils.py

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# SELECTIVITY (MIGHT BE BROKEN AT THE MOMENT TODO)
# regions = ['right ALM', 'left ALM', 'right Medulla', 'left Medulla']
regions = ['left ALM', 'right Medulla']
# regions = ['right ALM']

cond2use = [3,4] 

# determine pref direction based on these time points relative to alignEvent
tedges = [-0.5, -0.01]
# tedges = [-1.8, -1.2] # determine pref direction based on these time points relative to alignEvent
# tedges = [0.01, 0.5] # determine pref direction based on these time points relative to alignEvent
selCorr = np.zeros(
    (len(par.time), len(par.time), len(regions)))
for i,reg in enumerate(regions):
    # TODO - change this to calcPrefSelectivity
    sel = utils.calcSelectivity(psth, cond2use, region, tedges, params, par, pref_=1, plot=0)

    # utils.selectivity_heatmap(sel,params,par.time,np.arange(sel.shape[1]),(-2,2),(0,sel.shape[1]),hline=0)
    
    # fpth = os.path.join(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs',sub,date)
    # fname = 'populationSelectivity_' + region
    # utils.mysavefig(fpth,fname)

    # SELECTIVITY CORRELATION MATRIX
    # selCorr[:, :, i] = utils.calcSelectivityCorrelation(sel, params, par, region)

    # fpth = os.path.join(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs',sub,date)
    # fname = 'selectivityMatrix_' + region
    # utils.mysavefig(fpth,fname)
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# CODING DIRECTIONS
# regions = ['right ALM', 'left ALM', 'right Medulla', 'left Medulla']
regions = ['left ALM']
# regions = ['right ALM']

cond2use = dict()
cond2use['stimulus'] = [3,5,4,6] # (<RR> + <RL>) - (<LL> + <LR>)
cond2use['choice']   = [3,6,4,5] # (<RR> + <LR>) - (<LL> + <RL>)
cond2use['action']   = [3,6,4,5] # (<RR> + <LR>) - (<LL> + <RL>)
# cond2use['outcome']   = [3,4,5,6] # (<RR> + <RR>) - (<RL> + <LR>)

tedges = dict()
tedges['stimulus'] = [-1.85, -1.2] 
tedges['choice']   = [-0.5, 0] # (<RR> + <LR>) - (<LL> + <RL>) == (3+6)-(4+5)
tedges['action']   = [0, 0.5] # (<RR> + <LR>) - (<LL> + <RL>) == (3+6)-(4+5)
# tedges['outcome']   = [2, 2.4] # (<RR> + <LR>) - (<LL> + <RL>) == (3+6)-(4+5)

# determine pref direction based on these time points relative to alignEvent
# tedges = [-0.5, -0.01]
# tedges = [-1.8, -1.2] # determine pref direction based on these time points relative to alignEvent
# tedges = [0.01, 0.5] # determine pref direction based on these time points relative to alignEvent
selCorr = np.zeros(
    (len(par.time), len(par.time), len(regions)))


sproj = dict()  # single trials projs along stimulus mode
cproj = dict()  # single trials projs along choice mode
aproj = dict()  # single trials projs along action mode
for i,reg in enumerate(regions):
    # TODO - change this to calcPrefSelectivity
    # sel = utils.calcSelectivity(psth, cond2use, region, tedges, params, par, pref_=1, plot=0)

    # calculate coding directions
    dat = psth[reg]
    dat = (dat - np.mean(dat,axis=2)[:,:,np.newaxis]) #/ np.std(dat,axis=2)[:,:,np.newaxis]
    # for j in range(dat.shape[2]): # normalize within condition
    #     dat[:,:,j] = (dat[:,:,j] - dat[:,:,j].mean()) / (dat[:,:,j].std())

    stimulus = utils.codingDirection(dat, cond2use['stimulus'], tedges['stimulus'], params, par)
    choice = utils.codingDirection(dat, cond2use['choice'], tedges['choice'], params, par)
    action = utils.codingDirection(dat, cond2use['action'], tedges['action'], params, par)
    # outcome = utils.codingDirection(psth[reg], cond2use['outcome'], tedges['outcome'], params, par)
    
    CDs = np.vstack((stimulus,choice,action)).T
    Q , _ = np.linalg.qr(CDs)
    
    # stimulusproj = np.einsum('ijk,j->ik', psth[reg], Q[:,0])
    # choiceproj = np.einsum('ijk,j->ik', psth[reg], Q[:,1])
    # actionproj = np.einsum('ijk,j->ik', psth[reg], Q[:,2])
    stimulusproj = np.einsum('ijk,j->ik', dat, Q[:,0])
    choiceproj = np.einsum('ijk,j->ik', dat, Q[:,1])
    actionproj = np.einsum('ijk,j->ik', dat, Q[:,2])
    # outcomeproj =  np.einsum('ijk,j->ik', psth[reg], Q[:,3])
    
    cols = utils.Colors()
    cond2plot = [3,4,7,8]
    lw = [2,2,1,1]
    c = [cols.rhit,cols.lhit,cols.rmiss,cols.lmiss]
    fig,ax = plt.subplots(1,CDs.shape[1],figsize=(8,2),constrained_layout=True)
    for j in range(len(cond2plot)):
        cond = cond2plot[j]
        ax[0].plot(par.time,stimulusproj[:,cond],lw=lw[j],color=c[j])
        ax[1].plot(par.time,choiceproj[:,cond],lw=lw[j],color=c[j])
        ax[2].plot(par.time,actionproj[:,cond],lw=lw[j],color=c[j])
        # ax[3].plot(par.time,outcomeproj[:,cond],lw=lw[i],color=c[i])
    for j in range(len(ax)):
        ax[j].set_xlim((-2.5,2))
        for ev,evtm in params.ev.items():
            ax[j].axvline(evtm, color=(0,0,0), linestyle=(0, (1, 1)),linewidth=2.5)
    sns.despine(fig,ax,offset=0,trim=False)
    fig.supxlabel('Time from go cue (s)')
    fig.supylabel('Projection (a.u.)')
    fig.suptitle(reg)
    plt.show()
    
    # fpth = os.path.join(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs',sub,date)
    # fname = 'codingDirections' + reg
    # utils.mysavefig(fpth,fname)
    
    # single trial projections
    sproj[reg] = np.einsum('ijk,k->ij', trialdat[reg], Q[:,0])
    cproj[reg] = np.einsum('ijk,k->ij', trialdat[reg], Q[:,1])
    aproj[reg] = np.einsum('ijk,k->ij', trialdat[reg], Q[:,2])
    


# %% make a plot like susu's fig 7c using results from above cell
cond2use = [3,4]
reg2use = ['right ALM', 'left ALM']

trix = []
for i in cond2use:
    trix.append(params.trialid[i])

label = 'action'
# find time window 
tix = utils.findTimeIX(tedges[label],par.time)
tix = np.arange(tix[0],tix[1]+1)

x = cproj

xr0_c0_s = x[reg2use[0]][tix,:]
xr0_c0_s = np.mean(xr0_c0_s[:,trix[0]],axis=0) #* -1
xr1_c0_s = x[reg2use[1]][tix,:]
xr1_c0_s = np.mean(xr1_c0_s[:,trix[0]],axis=0)

xr0_c1_s = x[reg2use[0]][tix,:]
xr0_c1_s = np.mean(xr0_c1_s[:,trix[1]],axis=0) #* -1
xr1_c1_s = x[reg2use[1]][tix,:]
xr1_c1_s = np.mean(xr1_c1_s[:,trix[1]],axis=0)

s = 20
ec = 'w'
lw = 0.2
a = 0.9
fig,ax = plt.subplots(figsize=(3,2),constrained_layout=True)
ax.scatter(xr0_c0_s,xr1_c0_s,color=cols.rhit,s=s,edgecolors=ec,lw=lw,alpha=a)
ax.scatter(xr0_c1_s,xr1_c1_s,color=cols.lhit,s=s,edgecolors=ec,lw=lw,alpha=a)
sns.despine(fig,ax,offset=5,trim=False)
ax.set_xlabel(reg2use[0])
ax.set_ylabel(reg2use[1])
ax.set_title(label)
plt.show()

# fpth = os.path.join('/Users/munib/Economo-Lab/code/map-ephys/figs',sub,date)
# fname = 'pCD_' + label + '_' + reg2use[0] + 'vs' + reg2use[1]
# utils.mysavefig(fpth,fname)





# %% DLC DATA

acq = nwbfile.acquisition

feats = ['Camera0_side_TongueTracking',
         'Camera0_side_JawTracking',
         'Camera0_side_NoseTracking']

vidtm = acq['BehavioralTimeSeries'][feats[0]].timestamps[:]
viddt = stats.mode(np.diff(vidtm), keepdims=True)[0][0]
nFrames = len(vidtm)
thresh = 0.95

traj = np.zeros((nFrames, 2, len(feats)))
for i, feat in enumerate(feats):
    temp = acq['BehavioralTimeSeries'][feat].data[:][:, 0:2]
    proba = acq['BehavioralTimeSeries'][feat].data[:][:, 2]
    temp[proba < thresh, :] = np.nan
    traj[:, :, i] = temp


# %%
trialStart = trials_df.start_time
trialEnd = trials_df.stop_time
goStart = acq['BehavioralEvents']['go_start_times'].timestamps[:]


@widgets.interact(trial=widgets.IntSlider(0, min=0, max=params.nTrials-1), step=1)
def plotTimeSeries(trial):
    tstart = trialStart.iloc[trial]
    tend = trialEnd.iloc[trial]
    ix = utils.findTimeIX([tstart, tend], vidtm)
    ix = np.arange(ix[0], ix[1]+1)

    fig, ax = plt.subplots(2, 1, figsize=(7, 5))
    ax[0].plot(vidtm[ix], traj[ix, 1, 0])
    ax[0].axvline(goStart[trial], color=(0.5, 0.5, 0.5), linestyle='--')
    ax[1].plot(traj[ix, 0, 0], traj[ix, 1, 0])
    plt.show()


# %% PCA
from sklearn.decomposition import PCA

region = 'right Medulla'
cond2use = [3, 4]
dat = psth[region][:, :, cond2use]

datshape = dat.shape
datcat = np.concatenate((dat[:, :, 0], dat[:, :, 1]), axis=0)

pca = PCA(n_components=dat.shape[1])
X = pca.fit_transform(datcat)

X = np.einsum('ijk,cj->ick', dat, pca.components_)

colors = [(0, 0, 1, 0.7),
          (1, 0, 0, 0.7)]    # corresponding colors
fig, ax = plt.subplots(2, 1, figsize=(7, 5))
ax[0].plot(par.time, X[:,0,0],c=colors[0])
ax[0].plot(par.time, X[:,0,1],c=colors[1])
ax[1].plot(par.time, X[:,1,0],c=colors[0])
ax[1].plot(par.time, X[:,1,1],c=colors[1])
plt.show()


# %% GRAND AVERAGES

regions = ['right ALM', 'left ALM', 'right Medulla', 'left Medulla']
cols = ('b', 'r')


cond2use = [3, 4]
for region in regions:
    fig, ax = plt.subplots(figsize=(5, 3))
    for i in range(len(cond2use)):
        mu = np.mean(psth[region][:, :, cond2use[i]], 1)
        ci = np.std(psth[region][:, :, cond2use[i]], 1) / \
            np.sqrt(psth[region].shape[1])
        plt.plot(par.time, mu, c=cols[i])
        ax.fill_between(par.time, (mu-ci),
                        (mu+ci), color=cols[i], alpha=0.2, ec='none')
    plt.axvline(0, color=(0.5, 0.5, 0.5), linestyle='--')
    alignEvent = par.alignEvent
    plt.xlabel(f'Time from {alignEvent} (s)')
    plt.ylabel('Firing rate (spks/s)')
    ax.set_xlim(par.tmin, par.tmax)
    plt.title(region)
    plt.subplots_adjust(left=0.25, bottom=0.25)
    plt.show()

    # fpth = os.path.join(
    #     r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs', sub, date)
    # fname = 'grandAverageFiringRate_' + region
    # utils.mysavefig(fpth, fname)
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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



# %%
cols = np.array([[117,221,221],
                 [147,104,183],
                 [231,90,124],
                 [255,179,15],
                 [0,0,0]]) / 255
with plt.style.context('opinionated_rc'):
    # groups = units_df.groupby('brain_regions')
    groups = units_df.groupby('probe')
    fig,ax = plt.subplots(3,2,constrained_layout=True, figsize=(7,5))
    ax = ax.ravel()
    ct = 0
    for name, group in groups:
        # if name=='right Medulla':
        #     continue
        ax[ct].scatter(group.xpos,group.ypos,s=1,color=cols[ct],label=name,alpha=1)
        t = name + ', ' + np.array(group.ap_location)[0] + ' AP,' + np.array(group.ml_location)[0] + ' ML'
        t = np.unique(group.brain_regions)[0] + ', ' + np.array(group.ap_location)[0] + ' AP,' + np.array(group.ml_location)[0] + ' ML'
        ax[ct].set_title(t,fontsize=9,color=cols[ct])
        ct += 1
    for a in ax:
        a.set_xlim([-30, 900])
        a.tick_params(axis='both', which='major', labelsize=12)
    # plt.scatter(xpos,ypos,s=2)
    # ax.set_xlabel('xpos (um)')
    # ax.set_ylabel('ypos (um)')
    fig.supxlabel('xpos (um)',fontsize=10)
    fig.supylabel('ypos (um)',fontsize=10)
    plt.show()
    
fpth = os.path.join(r'C:\Users\munib\Documents\Economo-Lab\code\map-ephys\figs',sub,date)
fname = 'unit-locations-by-probe'
utils.mysavefig(fpth,fname)

# %%

