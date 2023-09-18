from scipy import stats
from scipy.signal.windows import gaussian
import os
from pynwb import NWBHDF5IO
import numpy as np
import pandas as pd

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
        units_df = addElectrodesInfoToUnits(units_df)
        # FIND CLUSTERS - subset units based on lowFR, quality, and region
        # params.cluid corresponds to units_df.iloc, not to units_df.unit
        # TODO - handle case where no cells found for a given region in par.regions
        units_df, params = findClusters(units_df, par, params)
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
def getBehavEventTimestamps(nwbfile, ev):
    # return timestamps of a behavioral event
    return nwbfile.acquisition['BehavioralEvents'][ev].timestamps[:]

# %%
def getEventTimes(nwbfile, trials_df, params, alignEvent, events):
    # returns params.ev = mode(behavioralEvent.timestamp) for 'events'
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
        evdict[ev] = np.round(stats.mode(evtm_aligned)[0][0],2)
        
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
def saveElectrodeCCFCoords(nwbfile,dataDir,sub,date):
    # get x,y,z coords in Allen CCF space for each unit/electrode
    units = nwbfile.units
    unit_electrodes = units.electrodes.data[:]


    posLabels = ['x','y','z']
    nUnits = len(nwbfile.units)
    coords = np.zeros((nUnits,3))
    probe = []
    probe_type = []
    electrode = []
    region = []
    for iunit in tqdm(range(nUnits)):
        for ilabel,label in enumerate(posLabels):
            # e = nwbfile.units[iunit]['electrodes'].item()
            coords[iunit,ilabel] = nwbfile.electrodes[unit_electrodes[iunit]][label].item()
        egroup_desc = units[iunit]['electrode_group'].item().description
        egroup_desc_dict = dict(eval(egroup_desc))
        probe.append(egroup_desc_dict['probe'])
        probe_type.append(egroup_desc_dict['probe_type']) 
        electrode.append(unit_electrodes[iunit])
        region.append(dict(eval(units[iunit].electrode_group.item().location))['brain_regions'])
    
    df = pd.DataFrame((coords),columns=posLabels)
    df['probe'] = probe
    df['probe_type'] = probe_type
    df['region'] = region
    df['electrode'] = electrode
    
    savedir = os.path.join(dataDir,'sub-'+sub)
    df.to_csv(os.path.join(savedir,'sub-' + sub + '_ses-' + date + '_ccfcoords.csv'),index=False)
    # coords = np.vstack((np.array(z),np.array(y),np.array(x))).T # (nUnits,3) # order to plot in brainrender
    # The common reference space is in PIR orientation where x axis = Anterior-to-Posterior, y axis = Superior-to-Inferior and z axis = Left-to-Right.
    # http://help.brain-map.org/display/mouseconnectivity/API#API-DownloadAtlas3-DReferenceModels
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
def addElectrodesInfoToUnits(units_df):
    # units_df - nwbfile.units.to_dataframe()
    # concatenates info stored in electrodes pynwb class to each row (unit) of the dataframe
    # this makes it easier to subset units based on brain region
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
def getSeq(nwbfile,par,params,units_df,trials_df):

    # TIME VECTOR
    edges = np.arange(par.tmin,par.tmax,par.dt)
    par.time = edges + par.dt / 2
    par.time = par.time[:-1]

    alignTimes = getBehavEventTimestamps(nwbfile,par.alignEvent)
    
    smN = par.smooth[0]
    smSTD = par.smooth[1]
    smBoundary = par.smooth[2]

    tstart = np.array(trials_df.start_time)
    tend = np.array(trials_df.stop_time)

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
            for trix,time in enumerate(alignTimes): # loop over trials and alignment times
                spktm = units_df.spike_times.iloc[unit]

                spktrial = findTrialForEvent(spktm, tstart, tend)
                # subset spktm to those that occur on current trial
                # spktm = spktm[spktrial==trix]

                spktm_aligned = spktm - time
                # Keep only spike times in a given time window around the stimulus onset
                trialtm = spktm_aligned[
                    (par.tmin < spktm_aligned) & (spktm_aligned < par.tmax)
                ]    
                N = np.histogram(trialtm, edges)
                trialdat[region][:,trix,iunit] = smooth(N[0] / par.dt,smN,smSTD,smBoundary).reshape(-1)
                
            # trial-average to get psth for each condition
            for icond in range(nCond):
                trix = params.trialid[icond]
                temp = trialdat[region][:,trix,iunit]
                psth[region][:,iunit,icond] = np.mean(temp,1) # mean across trials
    return trialdat,psth

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
def plotPSTH(trialdat,region,cond2plot,cols,lw,params,par,units_df,nwbfile,legend=None,plotRaster=0,plotWave=0):
    
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
        
        [ax[i].clear() for i in range(len(ax))]
        
        unitnum = units_df.iloc[unit].unit
        
        # PLOT RASTER
        if plotRaster:
            # setup raster yaxis
            tt = []
            nTrialsCond = [len(params.trialid[cond]) for i,cond in enumerate(cond2plot)]
            for i,nTrials in enumerate(nTrialsCond):
                tt.append(np.arange(nTrials))
            for i in range(1,len(cond2plot)):
                tt[i] = tt[i] + tt[i-1][-1] + 1
            # get spikes
            tm = units_df.iloc[unit].spike_times
            N = len(tm)
            # for each condition
            for i,cond in enumerate(cond2plot):
                trialtm_aligned = np.array([])
                trial = np.array([])
                trials = params.trialid[cond] 
                alignT = alignTimes[trials]
                # for each trial in condition
                for itrial,time in enumerate(alignT):
                    tm_aligned = tm - time
                    # Keep only spike times in a given time window around the stimulus onset
                    tm_aligned_trial = tm_aligned[
                        (par.tmin < tm_aligned) & (tm_aligned < par.tmax)
                    ]
                    trialtm_aligned = np.hstack((trialtm_aligned,tm_aligned_trial)) #.append(tm_aligned_trial)
                    trial = np.hstack((trial,[tt[i][itrial]]*len(tm_aligned_trial)))#.append([itrial]*len(tm_aligned_trial))

                ax[iraster].scatter(trialtm_aligned, trial, s=0.3, color=cols[i])
                ax[iraster].get_xaxis().set_ticklabels([])
                ax[iraster].set_xlim(par.tmin,par.tmax)
                ax[iraster].set_ylabel('Trials',fontsize=12)
                plotEventTimes(ax,params.ev)
        
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
            plotEventTimes(ax,params.ev)
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
        
        
# %% two functions in this cell are used in plotting heatmaps (see selectivity_heatmap())
def conv_index_to_bins(index):
    """Calculate bins to contain the index values.
    The start and end bin boundaries are linearly extrapolated from 
    the two first and last values. The middle bin boundaries are 
    midpoints.

    Example 1: [0, 1] -> [-0.5, 0.5, 1.5]
    Example 2: [0, 1, 4] -> [-0.5, 0.5, 2.5, 5.5]
    Example 3: [4, 1, 0] -> [5.5, 2.5, 0.5, -0.5]"""
    assert index.is_monotonic_increasing or index.is_monotonic_decreasing

    # the beginning and end values are guessed from first and last two
    start = index[0] - (index[1]-index[0])/2
    end = index[-1] + (index[-1]-index[-2])/2

    # the middle values are the midpoints
    middle = pd.DataFrame({'m1': index[:-1], 'p1': index[1:]})
    middle = middle['m1'] + (middle['p1']-middle['m1'])/2

    idx = pd.Index(middle,dtype='float64').union([start,end])
    # if isinstance(index, pd.DatetimeIndex):
    #     idx = pd.DatetimeIndex(middle).union([start,end])
    # elif isinstance(index, (pd.Float64Index,pd.RangeIndex,pd.Int64Index)):
    #     idx = pd.Index(middle,dtype='float64').union([start,end])
    # else:
    #     print('Warning: guessing what to do with index type %s' % 
    #         type(index))
    #     idx = pd.Float64Index(middle).union([start,end])

    return idx.sort_values(ascending=index.is_monotonic_increasing)

def calc_df_mesh(df):
    """Calculate the two-dimensional bins to hold the index and 
    column values."""
    return np.meshgrid(conv_index_to_bins(df.index), conv_index_to_bins(df.columns))


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
def plotEventTimes(ax,paramsev):
    # paramsev == params.ev
    for ev,evtm in paramsev.items(): # indicate epochs with vertical dashed lines
            ax.axvline(evtm, color=(0,0,0), linestyle=(0, (1, 1)),linewidth=2.5)