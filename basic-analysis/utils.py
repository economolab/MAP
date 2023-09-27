from scipy import stats
from scipy.signal.windows import gaussian
from scipy.interpolate import interp1d

import os
from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('dark_background')
# plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pitayasmoothie-light.mplstyle')
from ipywidgets import widgets
import opinionated as op
plt.style.use('default')
import matplotlib as mpl
mpl.rcParams['font.size'] = 12

from tqdm import tqdm

#from allensdk.core.reference_space_cache import ReferenceSpaceCache
from pathlib import Path
#import nrrd


# %% CLASSES
#####################################################################################################
#####################################################################################################
#####################################################################################################

# %%
class Colors:
    def __init__(self):
        self.rhit   = (0.08, 0.15, 0.72)
        self.lhit   = (0.72, 0.08, 0.13)
        self.rmiss  = (0.04, 0.07, 0.37)
        self.lmiss  = (0.37, 0.04, 0.04)
        
        self.alm       = (0.45882353, 0.86666667, 0.86666667)
        self.medulla   = (0.57647059, 0.40784314, 0.71764706)
        self.midbrain  = (0.90588235, 0.35294118, 0.48627451)
        self.striatum  = (1.        , 0.70196078, 0.05882353)
    # to return list of attribute names ( cols.__dict__.keys() )
    
# %%
# Turns a dictionary into a class
class Dict2Class(object):
    def __init__(self, my_dict):          
        for key in my_dict:
            setattr(self, key, my_dict[key])
            
# %%
def fieldnames(c):
    # returns attributes of a class
    # c is a class
    k = c.__dict__.keys()
    print('Attributes of class: ',list(k))
    return list(k)
    

# %% UTILITIES
#####################################################################################################
#####################################################################################################
#####################################################################################################

# %% 
def smooth(x, N, std, boundary=None):
    # % operates on first dimension only - make sure first dim is time
    # % gaussian kernel with window size N % TODO convert to standard deviation in time units
    # % boundary:
    # %   'reflect' - reflect N elements from beginning of time series across
    # %               0 index
    # %   'zeropad' - pad N zeros across 0 index
    # %   'none'    - don't handle boundary conditions

    # % returns:
    # % - out: filtered data, same size as x
    
    # ----------------------------------------------------------------------
    if N == 0 or N == 1: # no smoothing
        return x
    
    if len(x.shape) == 1: # if a vector (nRows,), turn into 2d array (nRows,1)
        x = x.reshape(-1,1)
    
    # handle boundary condition
    if boundary=='reflect':
        x_filt = np.concatenate((x[0:N+1,:],x) , axis=0)
        trim = N + 1
    elif boundary=='zeropad':
        x_filt = np.concatenate((np.zeros((N,x.shape[1])),x) ,axis=0)
        trim = N + 1
    elif boundary==None:
        x_filt = x
        trim = 0
    else:
        raise ValueError('UNKNOWN BOUNDARY TYPE')

    nRow = x_filt.shape[0]    
    nCol = x_filt.shape[1]

    kern = gaussian(N,std)
    kern[0:int(np.floor(N/2))] = 0; #causal
    kern = kern/np.sum(kern)
    
    out = np.zeros((nRow, nCol))
    for j in range(nCol):        
        out[:, j] = np.convolve(x_filt[:, j], kern, mode='same')
    
    out = out[trim:,:]
    
    return out

# %%
def permute(a,permute_idx):
    # a is a 2d numpy array, whose columns you want to permute
    # permute_idx is a list of the permutation you want
    idx = np.empty_like(permute_idx)
    idx[permute_idx] = np.arange(len(permute_idx))
    return a[:, idx]  # return a rearranged copy

# %%
def findTimeIX(tedges,time):
    # returns ix -> list of indicies corresponding to elements in time where tedges are found
    ix = []
    for t in tedges:
        ix.append((np.abs(time - t)).argmin())
    return ix

# %%
def renum(input_array):
    # https://stackoverflow.com/questions/62169627/renumber-a-sequence-to-remove-gaps-but-keep-identical-numbers
    # takes an array that has gaps in a sequence and removes gaps, but keeps duplicates.  
#     renum(np.array([1, 1, 1, 2, 2, 2]))  # already in correct shape
#     > [1, 1, 1, 2, 2, 2]

#     renum(np.array([1, 1, 2, 2, 4, 4, 5, 5, 5]))  # A jump between 2 and 4
#     > [1,1, 2, 2, 3, 3, 4, 4, 4]

#     renum(np.array([1, 1, 2, 2, 5, 2, 2]))  # A forward and backward jump
#     > [1,1, 2, 2, 3, 4, 4]
    diff = np.diff(input_array)
    diff[diff != 0] = 1
    renummed = np.hstack((input_array[0], diff)).cumsum()
    renummed_startAt1 = renummed - (renummed[0]-1)
    return renummed_startAt1


# %% FUNCTIONS FOR LOADING AND PROCESSING NWB FILES, SPECIFICALLY SUSU CHEN'S MAP DATA
#####################################################################################################
#####################################################################################################
#####################################################################################################

# %% 
def loadData(dataDir,sub,date,par,behav_only=0):
    # print details
    print(f"Loading data for sub-{sub}:{date} from {os.linesep}Data directory: {dataDir}")

    # find nwb file for given sub and date
    sessions = findNWB(dataDir, sub, date)
    # returns a list if more than one session provided, but this code only works for one session at the moment
    sessions = sessions[0] if len(sessions) == 1 else sessions 

    # LOAD DATA
    nwbfile = NWBHDF5IO(os.path.join(dataDir, "sub-"+sub, sessions)).read() # load nwb file

    # load coordinates
    coords_df = loadCoordsCSV(dataDir,sub,date)
    
    # to make data easier to use gonna convert some fields to pandas dataframes
    trials_df = nwbfile.trials.to_dataframe() 
    
    # FIND AND SUBSET SESSION-SPECIFIC DATA
    params = Dict2Class(dict())  # session specific parameters

    # FIND TRIALS FOR EACH CONDITION
    params.nTrials = trials_df.shape[0]
    params.trialid = findTrials(nwbfile, trials_df, par.condition)
    params.fs = stats.mode((nwbfile.units.sampling_rate.data[:]))[0] # neural data sampling frequency
    params.ev = getEventTimes(nwbfile, trials_df, params, par.alignEvent, par.events)

    # TIME VECTOR
    edges = np.arange(par.tmin,par.tmax,par.dt)
    par.time = edges + par.dt / 2
    par.time = par.time[:-1]

    # GET SINGLE TRIAL FIRING RATES & PSTHs
    if not behav_only:
        units_df = nwbfile.units.to_dataframe()
        units_df = addElectrodesInfoToUnits(units_df, coords_df)
        # FIND CLUSTERS - subset units based on lowFR, quality, and region
        # params.cluid corresponds to units_df.iloc, not to units_df.unit
        # TODO - handle case where no cells found for a given region in par.regions
        units_df, params = findClusters(units_df, par, params)
        units_df = alignSpikeTimes(nwbfile,units_df,trials_df,params,par)
        trialdat, psth = getSeq(nwbfile, par, params, units_df,trials_df)
    else:
        units_df = np.nan
        trialdat = np.nan
        psth = np.nan
        
    return nwbfile, units_df, trials_df, trialdat, psth, params


# %%
def findNWB(dataDir,sub,date):
    # dataDir - directory where data lives, should be where subject directories live
    # sub     - list of subjects
    # date    - list of dates for each subject
    # given a date corresponding to a session, returns the full name of the .nwb session file
    
    sessionList = os.listdir(os.path.join(dataDir,"sub-"+sub))
    sessions = [s for s in sessionList if date in s]
    sessions = [s for s in sessions if 'csv' not in s] # remove ccfcoords.csv from list
    return sessions

# %%
def loadCoordsCSV(dataDir,sub,date):
    sessionList = os.listdir(os.path.join(dataDir,"sub-"+sub))
    sessions = [s for s in sessionList if date in s]
    sessions = [s for s in sessions if 'csv' in s] # keep ccfcoords.csv from list
    coordsFile = sessions[0] if len(sessions) == 1 else sessions 

    if len(coordsFile) == 0:
        df = []
        print('No coordinates csv file found, using unregistered brain regions')
    else:
        df = pd.read_csv(os.path.join(dataDir,'sub-'+sub,coordsFile))

    return df

# %%
def findTrials(nwbfile,trials_df,conditions):
    # nwbfile - 
    # trials_df - nwbfile.trials.to_dataframe()
    # condition - list of query condition (example usage -> trials_df.query(condition))
    
    # get trial_uid for each condition
    trials = []
    for i in range(len(conditions)):
        cond = conditions[i]
        trials.append(trials_df.query(cond).trial_uid.to_numpy() - 1) # trial_uid is 1-indexed, so subtracting 1
    
    return trials

# %%
def getAllRegions(sub,date):
    if os.name == 'nt':  # Windows PC (office)
        dataDir = r'C:\Users\munib\Documents\Economo-Lab\code\map\meta'
    else:  # Macbook Pro M2
        # dataDir = '/Volumes/MUNIB_SSD/Economo-Lab/data/'
        dataDir = '/Users/munib/Economo-Lab/code/map/meta'
        
    df = pd.read_csv(os.path.join(dataDir,'allSessionsMetaData.csv'))
    
    regionstring = df[(df['sub'] == int(sub)) & (df['date'] == int(date))].regions
    
    samp= re.compile('[a-zA-z]+')
    word = samp.findall(regionstring.item())
    regions = []
    for i in range(1,len(word)-1,2):
        regions.append(word[i] + ' ' + word[i+1])
    
    return regions

# %%
def getBehavEventTimestamps(nwbfile, ev):
    # return timestamps of a behavioral event
    return nwbfile.acquisition['BehavioralEvents'][ev].timestamps[:]

# %%
def getEventTimes(nwbfile, trials_df, params, alignEvent, events):
    # returns params.ev = mode(behavioralEvent.timestamp) for 'events'
    # i just use the output of this to plot event markers, not much else
    tstart = np.array(trials_df.start_time)
    tstop = np.array(trials_df.stop_time)
    early = np.where(np.array(trials_df.early_lick == 'early'))

    t = np.vstack((tstart,tstop)).T # (trials,2)

    alignTimes = getBehavEventTimestamps(nwbfile, alignEvent)

    evdict = dict()

    for i,ev in enumerate(events):
        trix = []
        # which trial does each timestamp fall into
        evtm = getBehavEventTimestamps(nwbfile, ev)
        for itrix in range(len(evtm)):
            mask = np.logical_and(t[:,0]<evtm[itrix], t[:,1]>evtm[itrix])
            trix.append(int(mask.nonzero()[0]))
            
        
        align = alignTimes[trix]
        evtm_aligned = evtm - align
        evdict[ev] = np.round(stats.mode(evtm_aligned)[0],2)
    
        
    return evdict

# %%
def findTrialForEvent(ev, tstart, tend):
    # ev - np array times corresponding to an event
    # tstart - trial start times
    # tend = trial end times
    # returns trial - np array same size as ev with trial ev occurred on
    
    # ev = getBehavEventTimestamps(nwbfile, 'right_lick_times')
    # tstart = np.array(trials_df.start_time)
    # tend = np.array(trials_df.stop_time)
    
    # trial = np.zeros_like(ev)
    # for i in range(trial.shape[0]):
    #     t = np.where(np.logical_and(ev[i]>=tstart, ev[i]<=tend))[0].tolist()
    #     if len(t)==0:
    #         trial[i] = np.nan # probably occurred in the ITI
    #     else:
    #         trial[i] = t[0]

    # VECTORIZED VERSION - MUCH FASTER
    # Create a 2D boolean mask where each row represents
    # whether the corresponding event falls within a trial
    mask = (ev[:, None] >= tstart) & (ev[:, None] <= tend) # (nEvents,nTrials)
    
    # Find the first occurrence of True in each row
    first_occurrence = np.argmax(mask, axis=1)
    
    # Use this information to fill the 'trial' array
    trial = np.where(np.any(mask, axis=1), first_occurrence, np.nan)
    
    return trial
        
# %%
def saveCCFCoordsAndRegion(nwbfile,saveDir,ccfDir,sub,date):
    # ccfDir is path where allenccf data already lives / will be downloaded to if doesn't exist
    # code to get the annotation volume isn't here, I just downloaded it from:
    # http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/
    
    # get x,y,z coords in Allen CCF space for each unit/electrode
    units = nwbfile.units
    unit_id = units.unit.data[:]
    unit_electrodes = units.electrodes.data[:]
    
    # get allen ccf registered electrode coords
    electrodes = nwbfile.electrodes
    x = electrodes.x.data[:].astype(int) # ML in Allen CCF
    y = electrodes.y.data[:].astype(int) # DV in Allen CCF
    z = electrodes.z.data[:].astype(int) # AP in Allen CCF
    
    # get those coordinates for each unit
    x = x[unit_electrodes]
    y = y[unit_electrodes]
    z = z[unit_electrodes]
    
    # positions for electrodes outside brain are encoded as a very negative number. set these to 0
    x[x<0] = 0
    y[y<0] = 0
    z[z<0] = 0
    
    # save to dataframe
    df = pd.DataFrame()
    df['unit'] = unit_id
    df['x'] = x
    df['y'] = y 
    df['z'] = z 
    df['electrodes'] = unit_electrodes
    
    # get annotations
    annofile = 'annotation_10.nrrd' # 2017 - can download this using ReferenceSpaceCache.get_annotation_volume() or straight from server
    # http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/
    anno,header = nrrd.read(os.path.join(ccfDir, annofile))   
    # anno[0,:,:] coronal section
    # anno[:,0,:] dv section
    # anno[:,:,0] sagittal section
    
    # get reference space and name map (dictionary from structure id in annotations to name of structure)
    reference_space_key = os.path.join('annotation', 'ccf_2017')
    resolution = 10
    rspc = ReferenceSpaceCache(resolution, reference_space_key, manifest=Path(ccfDir) / 'manifest.json')
    # ID 1 is the adult mouse structure graph
    tree = rspc.get_structure_tree(structure_graph_id=1) 
    name_map = tree.get_name_map()
    
    # get structure ids for each unit/electrode
    struct_id = np.zeros_like(x)
    for i in range(len(struct_id)):
        pos = (np.array(df.iloc[i,1:4])[::-1] / resolution).astype(int) # get positions, divide by resolution, reverse to put in ccf coords            
        struct_id[i] = anno[pos[0],pos[1],pos[2]]

    region = [name_map[sid] if sid!=0 else '0' for sid in struct_id]
    acronym = [tree.get_structures_by_name([r])[0]['acronym'] if r!='0' else '0' for r in region]
    
    df['id'] = struct_id
    df['region'] = region
    df['acronym'] = acronym
    
    # add probe and probe_type
    nUnits = len(units)
    probe = []
    probe_type = []
    for iunit in range(nUnits):
        egroup_desc = units[iunit]['electrode_group'].item().description
        egroup_desc_dict = dict(eval(egroup_desc))
        probe.append(egroup_desc_dict['probe'])
        probe_type.append(egroup_desc_dict['probe_type']) 
        
    df['probe'] = probe
    df['probe_type'] = probe_type
    
    # save
    savedir = os.path.join(saveDir,'sub-'+sub)
    savefn = os.path.join(savedir,'sub-' + sub + '_ses-' + date + '_ccfcoords.csv')
    df.to_csv(savefn,index=False)
    print('Saved ' + savefn + ' !!!')
    
    # i = 0 
    # ix = (np.array(df.iloc[i,1:4])[::-1] / resolution).astype(int)
    # fig,ax = plt.subplots()
    # plt.imshow(anno[ix[0],:,:]) # plot coronal section
    # plt.clim(0,1500)
    # plt.set_cmap('gray')
    
    # ############### OLD 

    # posLabels = ['x','y','z']
    # nUnits = len(nwbfile.units)
    # coords = np.zeros((nUnits,3))
    # probe = []
    # probe_type = []
    # electrode = []
    # # region = []
    # for iunit in tqdm(range(nUnits)):
    #     for ilabel,label in enumerate(posLabels):
    #         # e = nwbfile.units[iunit]['electrodes'].item()
    #         coords[iunit,ilabel] = nwbfile.electrodes[unit_electrodes[iunit]][label].item()
    #     egroup_desc = units[iunit]['electrode_group'].item().description
    #     egroup_desc_dict = dict(eval(egroup_desc))
    #     probe.append(egroup_desc_dict['probe'])
    #     probe_type.append(egroup_desc_dict['probe_type']) 
    #     electrode.append(unit_electrodes[iunit])
    # #     region.append(dict(eval(units[iunit].electrode_group.item().location))['brain_regions'])
    
    # df = pd.DataFrame((coords),columns=posLabels)
    # df['unit'] = 
    # df['probe'] = probe
    # df['probe_type'] = probe_type
    # df['region'] = region
    # df['electrode'] = electrode
    
    # savedir = os.path.join(dataDir,'sub-'+sub)
    # df.to_csv(os.path.join(savedir,'sub-' + sub + '_ses-' + date + '_ccfcoords.csv'),index=False)
    
    # # The common reference space is in PIR orientation where x axis = Anterior-to-Posterior, y axis = Superior-to-Inferior and z axis = Left-to-Right.
    # # http://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas3-DReferenceModels
    
    return df

# %%
def taskPerformance(trials_df,trialid,cond2use, c, labels, plot=1):
    # trials_df - nwbfile.trials.to_dataframe()
    # trials - list that contains session-specific trials by condition
    # cond2use - np array of conditions to use corresponding to indices in trials
    perf = []
    trial_uid = trials_df['trial_uid'].to_numpy()
    for i in cond2use:
        trix = trialid[i] + 1 # params.trialid is 0-indexed, need to add 1 to match 1-indexing of trial_uid
        # find where trial_uid contains trix
        ix = np.where(np.in1d(trial_uid, trix))[0]
        df = trials_df.iloc[ix,:]
        nTrials = df.shape[0]
        nHitTrials = df.query('outcome == "hit"').shape[0]
        perf.append(nHitTrials / nTrials * 100)
    
    
    if plot:
        x = np.arange(0,len(perf)) # x-coordinates of your bars
        w = 0.8

        with plt.style.context('opinionated_rc'):
            fig, ax = plt.subplots(figsize=(2.5,3),constrained_layout=True)
            ax.bar(x,
                height=[np.mean(p) for p in perf],
                # yerr=[np.std(p) for p in perf],    # error bars
                width=w,    # bar width
                tick_label=labels,
                color=c,  # face color transparent
                edgecolor=c,
                # error_kw=dict(ecolor=(0.5,0.5,0.5), lw=2, capsize=5, capthick=2)
                )

            # for i in range(len(x)):
            #     # distribute scatter randomly across whole width of bar
            #     ax.scatter(x[i] + np.random.random(1) * w - w / 2, perf[i], color='k', edgecolors='w')
                
            ax.set_ylim((0,100))
            ax.set_ylabel('performance (%)')
            plt.show()    
        
        
    return perf


# %%
def getLickTimes(nwbfile, trials_df, par):
    # return lick times, lick direction, and corresponding trial
    
    tstart = np.array(trials_df.start_time)
    tend = np.array(trials_df.stop_time)
    alignTimes = getBehavEventTimestamps(nwbfile,par.alignEvent)
        
    # get lick times and trials
    ltm_right = getBehavEventTimestamps(nwbfile,'right_lick_times')
    trial_right = findTrialForEvent(ltm_right,tstart,tend)
    ltm_left  = getBehavEventTimestamps(nwbfile,'left_lick_times')    
    trial_left = findTrialForEvent(ltm_left,tstart,tend)
    
    # align and remove events/trials that occur outside trial (probably in the ITI)
    # they're set as nan in the output of findTrialForEvent()
    notNanEv = ~np.isnan(trial_right)
    ltm_right[notNanEv] = ltm_right[notNanEv] - \
        alignTimes[trial_right[notNanEv].astype(int)]
    notNanEv = ~np.isnan(trial_left)
    ltm_left[notNanEv] = ltm_left[notNanEv] - \
        alignTimes[trial_left[notNanEv].astype(int)]
        
    lickdir_right = np.zeros_like(ltm_right) # right licks = 0
    lickdir_left = np.zeros_like(ltm_left) + 1 # left licks = 1
    
    licktm = np.hstack((ltm_right,ltm_left))
    licktrial = np.hstack((trial_right,trial_left))
    lickdir = np.hstack((lickdir_right,lickdir_left))
    
    # sort in ascending trial order
    sortix = np.argsort(licktrial)
    lick = Dict2Class(dict())
    lick.tm = licktm[sortix]
    lick.trial = licktrial[sortix]
    lick.dir = lickdir[sortix]
    lick.dirdict = dict()
    lick.dirdict['right'] = 0
    lick.dirdict['left'] = 1
    
    return lick

# %%
def lickRaster(nwbfile, trials_df, par, params, cond2plot, cols, labels):
    # plot lick raster for provided conditions
    
    cols = np.array(cols)
    
    lick = getLickTimes(nwbfile, trials_df, par)
    
    trial = []
    tm = []
    dir = []
    c = []
    for i in range(len(cond2plot)):
        mask = np.isin(lick.trial,params.trialid[cond2plot[i]]) 
        trial.append( lick.trial[mask].astype(int) )
        trial[i] = renum(trial[i]) # remove gaps in sequence and start at 0
        tm.append( lick.tm[mask] )
        dir.append( lick.dir[mask] )
        c.append( cols[dir[i].astype(int),:] )


    # renumber trials so that conditions are plotted on top of each other
    for i in range(1,len(trial)):
        trial[i] += trial[i-1][-1] + 10
                
    s = 1
    with plt.style.context('opinionated_rc'):
        fig,ax = plt.subplots(constrained_layout=True,figsize=(4,4))
        for i in range(len(cond2plot)):
            # ax.scatter(tm[i],trial[i],color=c[i],s=s,label=labels[i])
            # ax.scatter(tm[i],trial[i],s=s,c=dir[i],cmap='seismic')
            ax.scatter(tm[i],trial[i],s=s,c=c[i],cmap='seismic')
            plotEventTimes(ax,params.ev)
        ax.set_xlabel(f'Time from {par.alignEvent} (s)',fontsize=12)
        ax.set_ylabel(f'Trials',fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        # op.add_legend(fontsize=12,markerscale=4)
    
    
    

# %%
def addElectrodesInfoToUnits(units_df,coords_df):
    # units_df - nwbfile.units.to_dataframe()
    # concatenates info stored in electrodes pynwb class if coords_df is not available
    # else concatenates coords_df
    # coords_df contains allenccf info about each electrode/unit

    if len(coords_df) == 0:
        nUnits = units_df.shape[0]
        electrodes_dict = []
        probe_dict = []
        for i in range(nUnits):
            estring = units_df.electrode_group.iloc[i].location
            electrodes_dict.append( dict(eval(estring)) )
            
            estring = units_df.electrode_group.iloc[i].description
            probe_dict.append( dict(eval(estring)) )
            
        electrodes_df = pd.DataFrame.from_records(electrodes_dict)
        probe_df = pd.DataFrame.from_records(probe_dict)
        
        units_df = pd.concat([units_df, electrodes_df, probe_df], axis=1)
    else:
        units_df = pd.concat([units_df, coords_df.iloc[:,1::]], axis=1)
        # rename 'acronym' column to 'brain_regions' -> already wrote code that uses brain_regions
        units_df.columns = ['brain_regions' if x=='acronym' else x for x in units_df.columns]


    return units_df

# %%
def findClusters(units_df,par,params):
    # units_df  - nwbfile.units.to_dataframe()
    # par - params used for all sessions
    # params    - params used for an individual session
    
    # subsets units_df to just units based on lowFR, quality, and brain region (all set in par)
    
    # subset by firing rate
    lowFR = par.lowFR
    # lowFRUnits = np.where(units_df.avg_firing_rate >= lowFR)[0]
    units_df.query('avg_firing_rate >= @lowFR',inplace=True)

    # find clusters that match par.quality
    quality = par.quality
    units_df.query('unit_quality == @quality',inplace=True)

    units_df = units_df.reset_index(drop=True)

    # find clusters that match params.regions
    params.cluid = dict()
    for region in par.regions:
        ix = np.where(units_df['brain_regions'].str.contains(region,case=False)) # unit index in units_df
        params.cluid[region] = ix[0]

    if len(params.cluid) == 1:
        params.allcluid = [v for k,v in params.cluid.items()]
        units_df = units_df.iloc[params.allcluid[0]]
    else:
        params.allcluid = np.hstack([v for k,v in params.cluid.items()])
        units_df = units_df.iloc[params.allcluid]

    # renumber cluid
    ct = 0
    for region in par.regions:
        N = len(params.cluid[region])
        params.cluid[region] = np.arange(ct,ct+N)
        ct += N

    units_df = units_df.reset_index(drop=True)
    
    return units_df, params

# %% 
def alignSpikeTimes(nwbfile,units_df,trials_df,params,par):
    # gets spike times within trial and then aligns to align event
    
    tstart = np.array(trials_df.start_time)
    tend = np.array(trials_df.stop_time)
    align = getBehavEventTimestamps(nwbfile,par.alignEvent)
    
    units_df['trial'] = np.nan
    units_df['trial'] = units_df['trial'].apply(lambda x: [x])
    units_df['trialtm'] = np.nan
    units_df['trialtm'] = units_df['trialtm'].apply(lambda x: [x])

    trial = []
    trialtm = []
    print('Getting trialtm_aligned and trial')
    for i in tqdm(range(units_df.shape[0])): #range(len(nwbfile.units)):
        tm = units_df['spike_times'][i]
        trial = findTrialForEvent(tm,tstart,tend).astype(int)
        trialtm = tm - align[trial]
        units_df['trial'][i] = trial
        units_df['trialtm'][i] = trialtm
        
    # print(units_df.iloc[0].trialtm.shape)
    # print(units_df.iloc[0].trial.shape)
    # print(units_df.iloc[0].spike_times.shape)    
    
    return units_df

    

    

# %%
def getSeq(nwbfile,par,params,units_df,trials_df):

    # TIME VECTOR
    edges = np.arange(par.tmin,par.tmax,par.dt)
    par.time = edges + par.dt / 2
    par.time = par.time[:-1]

    # alignTimes = getBehavEventTimestamps(nwbfile,par.alignEvent)
    
    smN = par.smooth[0]
    smSTD = par.smooth[1]
    smBoundary = par.smooth[2]

    tstart = np.array(trials_df.start_time)
    # tend = np.array(trials_df.stop_time)

    trialdat = dict()
    psth = dict()
    for region in par.regions:
        print(f'Getting neural data for {region}')

        nUnits = len(params.cluid[region])
        nCond = len(params.trialid)

        trialdat[region] = np.zeros( (len(par.time) , params.nTrials , nUnits)  ) # (time,trials,units)
        psth[region] = np.zeros( (len(par.time) , nUnits , nCond)  ) # (time,units,cond)

        
        for iunit in tqdm(range(nUnits)): # loop over units
            unit = params.cluid[region][iunit] 
            for trix in range(params.nTrials): # loop over trials and alignment times
                # get spikes in current trial
                trialtm = units_df.trialtm.iloc[unit] # already aligned
                trial = units_df.trial.iloc[unit]
                spkmask = np.isin(trial,trix)
                # if no spikes found for current set of trials (trix), move on to
                # next set
                if np.all(~spkmask):
                    continue

                # # Keep only spike times in a given time window around the stimulus onset
                # trialtm = spktm_aligned[
                #     (par.tmin < spktm_aligned) & (spktm_aligned < par.tmax)
                # ]    
                N = np.histogram(trialtm[spkmask], edges)
                trialdat[region][:,trix,iunit] = smooth(N[0] / par.dt,smN,smSTD,smBoundary).reshape(-1)
                
            # trial-average to get psth for each condition
            for icond in range(nCond):
                trix = params.trialid[icond]
                temp = trialdat[region][:,trix,iunit]
                psth[region][:,iunit,icond] = np.mean(temp,1) # mean across trials
                
    return trialdat,psth

# %%
def getBehavorialTimeSeries(nwbfile,feats,trials_df,par,thresh=0.95):
    # returns:
    # vidtm is video timestamp within trial
    # vidtrial is the trial each video timestamp is from 
    # traj is (time,coords,feature) -> (time,[x,y],[tongue,jaw,nose])
    # can get a trial's worth of data like:
    #   trialmask = vidtrial==0; vidtm0=vidtm[trialmask]; traj0=traj[trialmask,:,:]
    acq = nwbfile.acquisition

    vidtm = acq['BehavioralTimeSeries'][feats[0]].timestamps[:]
    # viddt = stats.mode(np.diff(vidtm)).mode[0]
    nFrames = len(vidtm)

    traj = np.zeros((nFrames, 2, len(feats))) # (time,coord,feats)
    for i, feat in enumerate(feats):
        temp = acq['BehavioralTimeSeries'][feat].data[:][:, 0:2]
        proba = acq['BehavioralTimeSeries'][feat].data[:][:, 2]
        temp[proba < thresh, :] = np.nan
        traj[:, :, i] = temp

    # find which trial each timestamp in vidtm belongs to
    tstart = np.array(trials_df.start_time)
    tend = np.array(trials_df.stop_time)
    align = getBehavEventTimestamps(nwbfile,par.alignEvent)
    vidtrial = findTrialForEvent(vidtm,tstart,tend)
    
    # there are timestamps that don't correspond to a trial, these 
    # are probably during the ITI.
    # we'll remove these from the data
    notnanmask = ~np.isnan(vidtrial)
    vidtrial = vidtrial[notnanmask].astype(int)
    vidtm_aligned = vidtm[notnanmask] - align[vidtrial] # also aligning vidtm
    traj = traj[notnanmask,:,:]

    # now we have the proper data, but just want to keep tmin to tmax within each trial
    timemask = (vidtm_aligned>=par.tmin) & (vidtm_aligned<=par.tmax)
    vidtm_aligned = vidtm_aligned[timemask]
    vidtrial = vidtrial[timemask]
    traj = traj[timemask,:,:]
    
    return vidtm_aligned,vidtrial,traj


def getFeatures(nwbfile,feats):
    # # get features to use in getTrajectories() given
    # feats = ['tongue','jaw','nose','etc']
    # and actual features are ['Cam*Jaw',etc]
    # and allfeats is just all those ['Cam*jaw,] features
    allfeats = list(nwbfile.acquisition['BehavioralTimeSeries'].time_series.keys())
    allfeats_lower = [f.lower() for f in allfeats]
    featmask = [i for i,x in enumerate(allfeats_lower) if any(x in y or y in x for y in feats)]
    newfeats = [allfeats[i] for i in featmask]
    newfeats_lower = [f.lower() for f in newfeats]
    # match newfeats to ordering in feats
    # featmask = [i for i,x in enumerate(newfeats) if any(x in y or y in x for y in feats)]
    
    ix = []
    for i in range(len(feats)):
        for j in range(len(newfeats_lower)):
            if feats[i] in newfeats_lower[j]:
                ix.append(j)

    feats = [newfeats[i] for i in ix]

    return feats,allfeats

# %%
def getTrajectories(nwbfile,feats,trials_df,par,params,thresh=0.95):
    # returns class traj
    # traj.ts is size (time,trials,coord,feat)
    # time axis is same as neural activity (tmin:dt:tmax)
    # traj.leg = feat and corresponds to feat axis in traj.ts
    
    outfeats,allfeats = getFeatures(nwbfile,feats)
        
    
    # get trajectories and with aligned video time and corresponding trials
    vidtm,vidtrial,traj = getBehavorialTimeSeries(nwbfile,outfeats,trials_df,par)
    
    traj_interp = np.nan * np.zeros(
        (len(par.time),params.nTrials,traj.shape[1],traj.shape[2])) # (time,trials,coord,feat)
    # for each trial
    for trix in range(params.nTrials):
        # get trial data
        mask = vidtrial == trix
        t = vidtrial[mask]
        ft = vidtm[mask]
        ts = traj[mask,:,:]
        
        # for each feature
        for ifeat in range(ts.shape[2]):
            traj_interp[:,trix,:,ifeat] = \
                interp1d(ft, ts[:,:,ifeat],axis=0,fill_value='extrapolate')(par.time)
                
        # find where trial starts, if it's before par.tmin, then need to fill nans before interp1d
        firstframetime = ft[0] # last frame time for current trial
        if firstframetime > par.tmin:
            firstframeix_interptime = findTimeIX([firstframetime],par.time)[0]
            traj_interp[0:firstframeix_interptime,trix,:,:] = np.nan
        
        # find where trial ends, if it's before par.tmax, then need to fill nans after interp1d
        lastframetime = ft[-1] # last frame time for current trial
        if lastframetime < par.tmax:
            lastframeix_interptime = findTimeIX([lastframetime],par.time)[0]
            traj_interp[lastframeix_interptime::,trix,:,:] = np.nan

    traj = Dict2Class(dict())
    traj.ts = traj_interp
    traj.leg = feats # corresponds to last dimension of ts
            
    return traj

# %%
def getKinematics(traj,feats):
    # returns class 'kin'
    # kin.ts = (time,trials,coord,feat)
    # kin.leg corresponds to feat axis
    pos = traj.ts # (time,trials,coord,feat)
    vel = np.gradient(pos,axis=0)

    kin = Dict2Class(dict())
    kin.ts = np.concatenate((pos,vel),axis=3) #(time,trials,coord,feat)
    kin.leg = [f+'_pos' for f in feats] + [f+'_vel' for f in feats]
    return kin

# %%
def getTrajAndKin(nwbfile,feats,trials_df,par,params,thresh=0.95):
    # threshold is for dlc likelihood
    traj = getTrajectories(nwbfile,feats,trials_df,par,params)
    kin = getKinematics(traj,feats)
    return traj,kin

# %% SELECTIVITY AND CODING DIRECTIONS
#####################################################################################################
#####################################################################################################
#####################################################################################################
# %%
def calcSelectivity(psth,cond2use,region,tedges,params,par,pref_=1,plot=1):
    dat = psth[region][:,:,cond2use]

    # find preferred direction
    tix = findTimeIX(tedges,par.time)
    tix = np.arange(tix[0],tix[1]+1)

    pref = np.argmax(np.mean(dat[tix,:], 0), 1)
    nonpref = pref.copy()
    temp = pref.copy()
    nonpref[temp==0] = 1
    nonpref[temp==1] = 0

    # calculate selectivity as pref-nonpref
    nUnits = pref.shape[0]
    sel = np.zeros( (len(par.time) , nUnits)  ) # (time,units)
    for i in range(pref.shape[0]):
        if pref_:
            sel[:,i] = dat[:,i,pref[i]] - dat[:,i,nonpref[i]]
        else:
            sel[:,i] = dat[:,i,0] - dat[:,i,1] # right - left
        


    if plot:
        # mean and ci    
        mu = np.mean(sel,1)
        ci = 1.96 * np.std(sel,1)/np.sqrt(nUnits)

        fig, ax = plt.subplots(figsize=(3,2))
        plt.rcParams['font.size'] = 14
        ax.plot(par.time,mu,color=(0.66, 0.41, 0.96))
        ax.fill_between(par.time, (mu-ci), (mu+ci), color=(0.66, 0.41, 0.96), alpha=0.2,ec='none')
        plotEventTimes(ax,params.ev)
        alignEvent = par.alignEvent
        plt.xlabel(f'Time from {alignEvent} (s)')
        plt.ylabel('Selectivity (spks/s)')
        plt.title(region)
        ax.set_xlim(par.tmin,par.tmax)
        y = ax.get_ylim()
        plt.fill_between([tedges[0],tedges[1]],y[0],y[1], color=np.array([137, 196, 214])/255,alpha=0.2, ec='none')
        ax.set_xlim(par.tmin,par.tmax)
        ax.set_ylim(y)
        # ax.set_facecolor("white")
        sns.despine(fig,ax,offset=0,trim=False)
        plt.subplots_adjust(left=0.25,bottom=0.25)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
            ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(10)


    return sel

# %%
def calcSelectivityCorrelation(sel,params,par,region,plot=1):
    n = sel.shape[0]
    c = np.zeros( (n,n) )
    for i in range(n):
        for j in range(n):
            a = np.corrcoef(sel[i,:],sel[j,:])
            c[i,j] = a[0,1]

    if plot:
        selectivity_heatmap(c,params,par.time,par.time,(-2,2),(-2,2))
        
    return c


def codingDirection(psth,cond2use,tedges,params,par):
    # cond2use should be in the following order -> (<C1> + <C2>) - (<C3> + <C4>)
    dat = psth[:,:,cond2use] # (time,units,conditions)
    
    # find time window 
    tix = findTimeIX(tedges,par.time)
    tix = np.arange(tix[0],tix[1]+1)

    # calculate CD
    nUnits = dat.shape[1]
    nTime = dat.shape[0]
    # cddat = np.zeros((nTime, nUnits)) # (time,units)
    
    mu = np.mean(dat[tix,:,:],axis=0) # (units,conditions)
    sd = np.std(dat[tix,:,:],axis=0) # (units,conditions)
    
    cd = (mu[:,0] + mu[:,1]) - (mu[:,2] + mu[:,3]) # (units,)
    cd = cd / np.sqrt(np.sum(np.square(sd),axis=1)) # divide by sum of variances across conditions
    
    cd = cd / np.sum(np.abs(cd))



    return cd

# %% PLOTTING
#####################################################################################################
#####################################################################################################
#####################################################################################################

# %%
def mysavefig(fpth,fname,dpi=300,format='png'):
    if not os.path.isdir(fpth):
        # os.mkdir(fpth)
        os.makedirs(fpth, exist_ok=True)
    plt.savefig(os.path.join(fpth,fname)+'.'+format, dpi=dpi, format=format)
    
# %%
def plotPSTH(trialdat,region,cond2plot,cols,lw,params,par,units_df,nwbfile,legend=None,plotRaster=0,plotWave=0,buffer=10):
    # buffer = how many 'trials' in between conditions in raster plot
    ## TODO - if plotting a photoinactivation condition, plot shaded region during stimulus period
    
    time = par.time
    nUnits = trialdat[region].shape[2]
    alignTimes = getBehavEventTimestamps(nwbfile,par.alignEvent)
    
    
    # SETUP AXES
    if plotRaster and plotWave:
        plt.style.use('opinionated_rc')
        fig, ax = plt.subplots(3,1, constrained_layout=True, figsize=(4,5),  gridspec_kw={'height_ratios': [1.5, 1.5, 1]})
        iraster = 0
        ipsth = 1
        iwave = 2
    elif plotRaster:
        plt.style.use('opinionated_rc')
        fig, ax = plt.subplots(2,1, constrained_layout=True, figsize=(4,4))
        iraster = 0
        ipsth = 1
    elif plotWave:
        plt.style.use('opinionated_rc')
        fig, ax = plt.subplots(2,1, constrained_layout=True, figsize=(4,4))
        ipsth = 0
        iwave = 1
    else:
        plt.style.use('default')
        fig, ax = plt.subplots(constrained_layout=True, figsize=(3,2))
        ipsth = 0
        ax = [ax]

    # WIDGET TO EASILY LOOK AT UNITS    
    widgUnit = widgets.IntSlider(0, min=0, max=nUnits-1)    
    @widgets.interact(unit=widgUnit, step=1)
    def plotRasterAndPSTHWidget(unit):
        
        # # SETUP AXES
        # if plotRaster and plotWave:
        #     plt.style.use('opinionated_rc')
        #     fig, ax = plt.subplots(3,1, constrained_layout=True, figsize=(4,5),  gridspec_kw={'height_ratios': [1.5, 1.5, 1]})
        #     iraster = 0
        #     ipsth = 1
        #     iwave = 2
        # elif plotRaster:
        #     plt.style.use('opinionated_rc')
        #     fig, ax = plt.subplots(2,1, constrained_layout=True, figsize=(4,4))
        #     iraster = 0
        #     ipsth = 1
        # elif plotWave:
        #     plt.style.use('opinionated_rc')
        #     fig, ax = plt.subplots(2,1, constrained_layout=True, figsize=(4,4))
        #     ipsth = 0
        #     iwave = 1
        # else:
        #     plt.style.use('default')
        #     fig, ax = plt.subplots(constrained_layout=True, figsize=(3,2))
        #     ipsth = 0
        #     ax = [ax]
    
        
        [ax[i].clear() for i in range(len(ax))]
        
        iunit = params.cluid[region][unit]
        unitnum = units_df.iloc[iunit].unit
        
                
        # PLOT RASTER
        if plotRaster:
            # get spikes
            trialtm = units_df.iloc[iunit].trialtm # already aligned
            trial = units_df.trial.iloc[iunit]
            # for each condition
            lasttrial = 0
            for i,cond in enumerate(cond2plot):
                trials = params.trialid[cond] 
                spkmask = np.isin(trial,trials)
                spks2plot = trialtm[spkmask]
                trials2plot = renum(trial[spkmask]) + lasttrial + buffer
                lasttrial = trials2plot[-1] + 1
                ax[iraster].scatter(spks2plot, trials2plot, s=0.3, color=cols[i])
                ax[iraster].get_xaxis().set_ticklabels([])
                ax[iraster].set_xlim(par.tmin,par.tmax)
                ax[iraster].set_ylabel('Trials',fontsize=12)
                
            plotEventTimes(ax[iraster],params.ev)
        
        # PLOT PSTH
        for i,cond in enumerate(cond2plot):
            
            t = params.trialid[cond]
            dat = trialdat[region][:,t,unit]
            mu = np.mean(dat,axis=1) # mean across trials in condition
            sem = np.std(dat,axis=1) / np.sqrt(len(t)) # std err of mean across trials in condition
            if legend: 
                leg = legend[i]
            else:
                leg = '_nolegend_'
            ax[ipsth].plot(par.time,mu,color=cols[i],lw=lw[i], label=leg)
            # ax[ipsth].fill_between(par.time, (mu-sem), (mu+sem), 
            #                 color=cols[i], alpha=0.2,ec='none', label='_nolegend_')
            ax[ipsth].fill_between(par.time, (mu-sem), (mu+sem), 
                            color=cols[i], alpha=0.2,ec='none', label=leg)
            ax[ipsth].set_xlabel(f'Time from {par.alignEvent} (s)',fontsize=12)
            ax[ipsth].set_ylabel('Firing rate (spks/s)',fontsize=12)
            ax[ipsth].set_xlim(par.tmin,par.tmax)
            ax[ipsth].grid(False)
            sns.despine(ax=ax[ipsth],offset=0,trim=False)
            # if legend:
            #     op.add_legend(fontsize=10) # NOT WORKING ANYMORE? TODO
        plotEventTimes(ax[ipsth],params.ev)
            
        # PLOT WAVE   
        if plotWave:
            wv = units_df.iloc[iunit].waveform_mean
            nSamples = len(wv)
            x = np.arange(nSamples) / units_df.iloc[iunit].sampling_rate * 1000
            ax[iwave].plot(x,wv,lw=3,color='k')      
            ax[iwave].set_xlabel('Time (ms)',fontsize=12)
            ax[iwave].set_ylabel('uV',fontsize=12)  
                
        with plt.style.context('default'):
            ax[0].set_title(f'{region} - Unit {unitnum}',fontsize=12)
        fig.canvas.toolbar_position = 'bottom'
        # ax.tick_params(direction='out', length=6, width=1)

# %% plot a single unit (index 1 of psth[region])
def plotSinglePSTH(unit,trialdat,region,cond2plot,cols,lw,params,par,units_df,nwbfile,legend=None,plotRaster=0,plotWave=0,buffer=10):
    # same as plotPSTH, but without unit slider, so you can look at stuff without setting that up if you don't want o
    ## TODO - if plotting a photoinactivation condition, plot shaded region during stimulus period
    
    time = par.time
    nUnits = trialdat[region].shape[2]
    alignTimes = getBehavEventTimestamps(nwbfile,par.alignEvent)
    
    # SETUP AXES
    if plotRaster and plotWave:
        plt.style.use('opinionated_rc')
        fig, ax = plt.subplots(3,1, constrained_layout=True, figsize=(4,5),  gridspec_kw={'height_ratios': [1.5, 1.5, 1]})
        iraster = 0
        ipsth = 1
        iwave = 2
    elif plotRaster:
        plt.style.use('opinionated_rc')
        fig, ax = plt.subplots(2,1, constrained_layout=True, figsize=(4,4))
        iraster = 0
        ipsth = 1
    elif plotWave:
        plt.style.use('opinionated_rc')
        fig, ax = plt.subplots(2,1, constrained_layout=True, figsize=(4,4))
        ipsth = 0
        iwave = 1
    else:
        plt.style.use('default')
        fig, ax = plt.subplots(constrained_layout=True, figsize=(3,2))
        ipsth = 0
        ax = [ax]
    
            
    unitnum = units_df.iloc[unit].unit
    
    # PLOT RASTER
    if plotRaster:
        # get spikes
        trialtm = units_df.iloc[unit].trialtm # already aligned
        trial = units_df.trial.iloc[unit]
        # for each condition
        lasttrial = 0
        for i,cond in enumerate(cond2plot):
            trials = params.trialid[cond] 
            spkmask = np.isin(trial,trials)
            spks2plot = trialtm[spkmask]
            trials2plot = renum(trial[spkmask]) + lasttrial + buffer
            lasttrial = trials2plot[-1] + 1
            ax[iraster].scatter(spks2plot, trials2plot, s=0.3, color=cols[i])
            ax[iraster].get_xaxis().set_ticklabels([])
            ax[iraster].set_xlim(par.tmin,par.tmax)
            ax[iraster].set_ylabel('Trials',fontsize=12)
            
        plotEventTimes(ax[iraster],params.ev)
    
    # PLOT PSTH
    for i,cond in enumerate(cond2plot):
        
        t = params.trialid[cond]
        dat = trialdat[region][:,t,unit]
        mu = np.mean(dat,axis=1) # mean across trials in condition
        sem = np.std(dat,axis=1) / np.sqrt(len(t)) # std err of mean across trials in condition
        if legend: 
            leg = legend[i]
        else:
            leg = '_nolegend_'
        ax[ipsth].plot(par.time,mu,color=cols[i],lw=lw[i], label=leg)
        ax[ipsth].fill_between(par.time, (mu-sem), (mu+sem), 
                        color=cols[i], alpha=0.2,ec='none', label='_nolegend_')
        plotEventTimes(ax[ipsth],params.ev)
        ax[ipsth].set_xlabel(f'Time from {par.alignEvent} (s)',fontsize=12)
        ax[ipsth].set_ylabel('Firing rate (spks/s)',fontsize=12)
        ax[ipsth].set_xlim(par.tmin,par.tmax)
        ax[ipsth].grid(False)
        sns.despine(ax=ax[ipsth],offset=0,trim=False)
        if legend:
            op.add_legend(fontsize=10)
        
    # PLOT WAVE   
    if plotWave:
        wv = units_df.iloc[unit].waveform_mean
        nSamples = len(wv)
        x = np.arange(nSamples) / units_df.iloc[unit].sampling_rate * 1000
        ax[iwave].plot(x,wv,lw=3,color='k')      
        ax[iwave].set_xlabel('Time (ms)',fontsize=12)
        ax[iwave].set_ylabel('uV',fontsize=12)  
            
    with plt.style.context('default'):
        ax[0].set_title(f'{region} - Unit {unitnum}',fontsize=12)
    fig.canvas.toolbar_position = 'bottom'
    # ax.tick_params(direction='out', length=6, width=1)

# %%
def plotSingleTrialTrajectories(time,traj,params,par,coords2plot,feats2plot):
    fig,ax = plt.subplots(figsize=(4,3), constrained_layout=True)
    @widgets.interact(trial=widgets.IntSlider(0, min=0, max=params.nTrials-1), step=1)
    def plotTraj(trial):
        ax.clear()
        with plt.style.context('opinionated_rc'):
            for i in range(len(feats2plot)):
                ax.plot(time,traj[:,trial,coords2plot,feats2plot[i]])
            # ax.plot(time,traj[:,trial,coords2plot,1])
            # ax.plot(time,traj[:,trial,coords2plot,2])

            plotEventTimes(ax,params.ev)
            ax.set_xlabel('Time from ' + par.alignEvent + ' (s)')
            ax.set_ylabel('Pixel')
            ax.set_xlim((par.tmin,par.tmax))
            sns.despine(ax=ax,offset=0,trim=False)
        

# %%
def selectivity_heatmap(dat,params,rows,cols,xlim,ylim,cmap='coolwarm',vline=1,hline=1):
    """Plot a heatmap of the dataframe values using the index and 
    columns"""
    df = pd.DataFrame(dat,columns=cols,index=rows)
    X,Y = calc_df_mesh(df)
    
    X = X[0:-1,0:-1].T
    Y = Y[0:-1,0:-1].T
    
    print(X.shape)
    print(Y.shape)
    print(df.shape)
    
    fig, ax = plt.subplots()
    c = plt.pcolormesh(X, Y, df, cmap=cmap)
    plt.colorbar(c)
    for ev,evtm in params.ev.items():
        if vline:
            plt.axvline(evtm, color=(0,0,0), ls=(0, (1, 1)),lw=1.5)
        if hline:
            plt.axhline(evtm, color=(0,0,0), ls=(0, (1, 1)),lw=1.5)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.show()

# %%
def plotEventTimes(ax,paramsev,color=(0,0,0),lw=2):
    # paramsev == params.ev
    for ev,evtm in paramsev.items(): # indicate epochs with vertical dashed lines
            ax.axvline(evtm, color=color, linestyle=(0, (1, 1)),linewidth=lw)


# %%
def plotKin(p,par,params,buffer=1):
    # buffer is spacing between trials when 'stacked'

    # get index of feature to plot
    ifeat = [i for i,f in enumerate(p.dat.leg) if p.feat in f][0]

    if p.style=='stacked':
        with plt.style.context('opinionated_rc'):
            fig,ax = plt.subplots(figsize=(4,8),constrained_layout=True)
            ct = 0
            for i,cond in enumerate(p.cond2plot):
                trials = params.trialid[cond]
                for j,trix in enumerate(trials):
                    toplot = p.dat.ts[:,trix,p.coord,ifeat]
                    ax.plot(par.time,toplot + ct*buffer, 
                            color=p.cols[i], alpha=p.alpha)
                    ct += 1
            sns.despine(fig,ax,offset=0,trim=False)
            plotEventTimes(ax,params.ev)
            ax.set_xlabel('Time from ' + par.alignEvent,fontsize=10)
            ax.set_yticklabels('')
            ax.set_xlim((par.tmin,par.tmax))
            plt.title(p.feat,fontsize=10)
            plt.show()
    elif p.style=='heatmap':
        with plt.style.context('opinionated_rc'):
            fig,ax = plt.subplots(figsize=(4,5),constrained_layout=True)
            nTrials = []
            dat = []
            for i,cond in enumerate(p.cond2plot):
                trials = params.trialid[cond]
                nTrials.append(len(trials))
                dat.append(p.dat.ts[:,trials,p.coord,ifeat])
            alldat = dat[0]
            for i in range(1,len(dat)):
                alldat = np.concatenate((alldat,dat[i]),axis=1)
            
            # converting data to dataframe and setting index to time.
            #  This is the only way I've figured out so far to get the time axis in 
            # a heatmap to be somewhat nice. 
            df = pd.DataFrame(alldat, index=np.round(par.time,1))
            hm = sns.heatmap(df.T,ax=ax,cmap='viridis',cbar=True)
            hm.set_facecolor('black') # nans are not visible, so set a background color for nans
            ax.set_xlabel('Time from ' + par.alignEvent,fontsize=10)
            ax.set_ylabel('Trials',fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.collections[0].colorbar.ax.tick_params(labelsize=7)
            # plot horizontal lines to mark different conditions
            for i in range(1,len(nTrials)):
                ax.axhline(nTrials[i], color=(1,1,1), linestyle=(0, (1, 1)),linewidth=0.75)
            plt.title(p.feat,fontsize=10)
            xlims = [par.tmin,par.tmax]
            # for heatmaps, have to use index, not time, for plotting events
            ix = findTimeIX(xlims,par.time)
            ax.set_xlim((ix[0],ix[1]))
            # same idea applies for the event times
            ev = params.ev.copy()
            for k,v in ev.items():
                ev[k] = findTimeIX([v],par.time)[0]
            plotEventTimes(ax,ev,color=(1,1,1),lw=1.5)
            plt.show()