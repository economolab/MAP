import os
import pandas as pd

def loadCoordinates(dataDir,sub,date):
    sessionList = os.listdir(os.path.join(dataDir,"sub-"+sub))
    sessions = [s for s in sessionList if date in s]
    sessions = [s for s in sessions if 'csv' in s] # keep ccfcoords.csv from list
    coordsFile = sessions[0] if len(sessions) == 1 else sessions 

    df = pd.read_csv(os.path.join(dataDir,'sub-'+sub,coordsFile))
    
    return df