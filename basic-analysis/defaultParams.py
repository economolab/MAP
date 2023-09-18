from utils import Dict2Class

def getDefaultParams():

    defparams = Dict2Class(dict())  # parameters for all sessions

    # BEHAVIORAL EVENTS
    defparams.alignEvent = 'go_start_times'
    defparams.events = [ # list of events to extract timestamp for (stored in params['ev'])
        # 'presample_start_times',
        'sample_start_times',
        'delay_start_times',
        'go_start_times'
    ]
    # events = [
    #     'delay_start_times',
    #     'delay_stop_times',
    #     'go_start_times',
    #     'go_stop_times',
    #     'left_lick_times',
    #     'photostim_start_times',
    #     'photostim_stop_times',
    #     'presample_start_times',
    #     'presample_stop_times',
    #     'right_lick_times',
    #     'sample_start_times',
    #     'sample_stop_times',
    #     'trialend_start_times',
    #     'trialend_stop_times',
    # ]

    # UNITS
    # remove clusters with a mean firing rates across all trials less than this value
    defparams.lowFR = 1
    defparams.quality = ['good']  # ['good','multi']; # unit qualities to use
    # ['right ALM','left ALM','right Medulla','left Medulla'] ## TODO: handle sessions with both hemis of a region
    defparams.regions = ['right ALM']

    # spike bins
    defparams.tmin = -2.5 # relative to alignEvent
    defparams.tmax = 2.5
    defparams.dt = 1/100
    defparams.smooth = (20,3,'reflect') # (N,std,boundary)) for causal gaussian filter

    # set conditions to calculate PSTHs for
    hit = 'outcome == "hit"'
    miss = 'outcome == "miss"'
    no = 'outcome == "ignore"'
    R = 'trial_instruction == "right"'
    L = 'trial_instruction == "left"'
    no_early = 'early_lick == "no early"'
    no_autowater = 'auto_water == 0'
    no_freewater = 'free_water == 0'
    no_stim = 'photostim_duration == "N/A"'
    stim = 'photostim_duration != "N/A"'


    # TRIAL TYPES / CONDITIONS
    defparams.condition = []
    # (0) not early, no auto/free water, no stim
    defparams.condition.append(
        no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
    # (1) right, not early, no auto/free water, no stim
    defparams.condition.append(
        R + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
    # (2) left, not early, no auto/free water, no stim
    defparams.condition.append(
        L + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
    # (3) right hits, not early, no auto/free water, no stim
    defparams.condition.append(
        hit + '&' + R + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
    # (4) left hits, not early, no auto/free water, no stim
    defparams.condition.append(
        hit + '&' + L + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
    # (5) right miss, not early, no auto/free water, no stim
    defparams.condition.append(
        miss + '&' + R + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
    # (6) left miss, not early, no auto/free water, no stim
    defparams.condition.append(
        miss + '&' + L + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
    # (7) right, no auto/free water, stim
    defparams.condition.append(
        R + '&' + no_autowater + '&' + no_freewater + '&' + stim)
    # (8) left, no auto/free water, stim
    defparams.condition.append(
        L + '&' + no_autowater + '&' + no_freewater + '&' + stim)
    # (9) right, no auto/free water, no stim
    defparams.condition.append(
        R + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
    # (10) left, no auto/free water, no stim
    defparams.condition.append(
        L + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
    

    defparams.behav_only = 0 # 1-trialdat,psth,units_df=NaN, 0-preprocess neural data

    
    return defparams


#     # BEHAVIORAL EVENTS
#     defparams['alignEvent'] = 'go_start_times'
#     defparams['events'] = [ # list of events to extract timestamp for (stored in params['ev'])
#         'presample_start_times',
#         'sample_start_times',
#         'delay_start_times',
#         'go_start_times'
#     ]
#     # events = [
#     #     'delay_start_times',
#     #     'delay_stop_times',
#     #     'go_start_times',
#     #     'go_stop_times',
#     #     'left_lick_times',
#     #     'photostim_start_times',
#     #     'photostim_stop_times',
#     #     'presample_start_times',
#     #     'presample_stop_times',
#     #     'right_lick_times',
#     #     'sample_start_times',
#     #     'sample_stop_times',
#     #     'trialend_start_times',
#     #     'trialend_stop_times',
#     # ]

#     # UNITS
#     # remove clusters with a mean firing rates across all trials less than this value
#     defparams['lowFR'] = 1
#     defparams['quality'] = ['good']  # ['good','multi']; # unit qualities to use
#     # ['right ALM','left ALM','right Medulla','left Medulla'] ## TODO: handle sessions with both hemis of a region
#     defparams['regions'] = ['right ALM']

#     # spike bins
#     defparams['tmin'] = -2.5 # relative to alignEvent
#     defparams['tmax'] = 2.5
#     defparams['dt'] = 1/100
#     defparams['smooth'] = (20,3,'reflect') # (N,std,boundary))

#     # set conditions to calculate PSTHs for
#     hit = 'outcome == "hit"'
#     miss = 'outcome == "miss"'
#     no = 'outcome == "ignore"'
#     R = 'trial_instruction == "right"'
#     L = 'trial_instruction == "left"'
#     no_early = 'early_lick == "no early"'
#     no_autowater = 'auto_water == 0'
#     no_freewater = 'free_water == 0'
#     no_stim = 'photostim_duration == "N/A"'
#     stim = 'photostim_duration != "N/A"'


#     # TRIAL TYPES / CONDITIONS
#     defparams.condition = []
#     # (0) not early, no auto/free water, no stim
#     defparams.condition.append(
#         no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
#     # (1) right, not early, no auto/free water, no stim
#     defparams.condition.append(
#         R + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
#     # (2) left, not early, no auto/free water, no stim
#     defparams.condition.append(
#         L + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
#     # (3) right hits, not early, no auto/free water, no stim
#     defparams.condition.append(
#         hit + '&' + R + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
#     # (4) left hits, not early, no auto/free water, no stim
#     defparams.condition.append(
#         hit + '&' + L + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
#     # (5) right miss, not early, no auto/free water, no stim
#     defparams.condition.append(
#         miss + '&' + R + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
#     # (6) left miss, not early, no auto/free water, no stim
#     defparams.condition.append(
#         miss + '&' + L + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + no_stim)
#     # (7) right, not early, no auto/free water, stim
#     defparams.condition.append(
#         R + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + stim)
#     # (8) left, not early, no auto/free water, stim
#     defparams.condition.append(
#         L + '&' + no_early + '&' + no_autowater + '&' + no_freewater + '&' + stim)