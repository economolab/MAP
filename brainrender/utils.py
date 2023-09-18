import os
import pandas as pd
import numpy as np

# %%
def loadCoordinates(dataDir,sub,date):
    sessionList = os.listdir(os.path.join(dataDir,"sub-"+sub))
    sessions = [s for s in sessionList if date in s]
    sessions = [s for s in sessions if 'csv' in s] # keep ccfcoords.csv from list
    coordsFile = sessions[0] if len(sessions) == 1 else sessions 

    df = pd.read_csv(os.path.join(dataDir,'sub-'+sub,coordsFile))
    
    return df

# %%
def permute(a,permute_idx):
    # a is a 2d numpy array, whose columns you want to permute
    # permute_idx is a list of the permutation you want
    idx = np.empty_like(permute_idx)
    idx[permute_idx] = np.arange(len(permute_idx))
    return a[:, idx]  # return a rearranged copy