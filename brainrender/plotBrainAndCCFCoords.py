

# %%
# https://github.com/brainglobe/brainrender/blob/master/examples/notebook_workflow.ipynb
# https://github.com/brainglobe/brainrender/blob/master/examples/add_cells.py

import os
import sys
import importlib
import random
import numpy as np
import pandas as pd

from brainrender import Scene, Animation
from vedo import embedWindow, Plotter, show  # <- this will be used to render an embedded scene 
from brainrender.actors import Points, PointsDensity, Cylinder

from rich import print
from myterial import orange
from pathlib import Path

import utils

# %%
_ = importlib.reload(sys.modules['utils'])

# %%

osname = os.name
if osname == 'nt':  # Windows PC (office)
    dataDir = r'C:\Users\munib\Documents\Economo-Lab\data'
else:  # Macbook Pro M2
    # dataDir = '/Volumes/MUNIB_SSD/Economo-Lab/data/'
    dataDir = '/Users/munib/Economo-Lab/data'

proj = "map" # subdirectory of dataDir
dataDir = os.path.join(dataDir, proj)

# sub = '484676' # subject/animal id
# date = '20210420' # session date
sub = '479121' # subject/animal id
date = '20200924' # session date

df = utils.loadCoordinates(dataDir,sub,date)

# %%

# ew = 'k3d' # render in interactive plot in vscode
ew = None  # render in separate window outside of jupyer notebook
embedWindow(ew)

title = 'sub-'+sub+'_ses-'+date
scene = Scene(title=title,atlas_name="allen_mouse_25um", inset=False, alpha=0.05)
if ew == 'k3d': 
    scene.jupyter = True

# reg2plot = ['AIp6a', 'B', 'CENT', 'CENT2', 'CENT3',  'CP',
#         'DT', 'FRP6a', 'FRP6b', 'I5', 'IC', 'ICc', 'ISN',
#         'LAV', 'LC', 'MEV', 'MOs1', 'MOs2/3', 'MOs5',
#        'ORBm6b', 'ORBvl6b', 'PIL', 'PL6a', 'POL', 'SGN', 'SIM', 'SLC',
#        'SLD', 'SNr', 'SPVO', 'SSp-ul6a', 'VIIn', 'bsc', 'cbf', 'dhc',
#         'gVIIn', 'scp']

reg2plot = ['AMBv',
 'V4',
 'LIN',
 'SPIV',
 'MY-sen',
 'MDRN',
 'SPVC',
 'ORBl6a',
 'ORBl6b',
 'CBX',
 'ORBvl6a',
 'ORBl5',
 'MOs1',
 'arb',
 'x',
 'aco',
 'ECU',
 'LRNm',
 'MOs2/3',
 'COPY',
 'MDRNd',
 'FRP6a',
 'FRP6b']
reg = []
for i in range(len(reg2plot)):
    a = scene.add_brain_region(reg2plot[i], alpha=0.15)
    # can speicfy color using hex (or a string)
    # a = scene.add_brain_region(reg2plot[i],color='#ff5733', alpha=0.15)
    # scene.add_label(a, reg2plot[i]) # only works with ew=None
# scene.add_brain_region('IRN',color='yellow', alpha=0.15)
# # mos = scene.add_brain_region("MOs",color='yellow', alpha=0.15)


# scene.add_label(irn, "IRN") # only works with ew=None
# scene.add_label(mos, "MOs")

# Add to scene
# plot electrodes of each probe separately (only way to color them separately)
groups = df.groupby('probe')
for p,group in groups:
    coords = np.array(group.iloc[:,0:3])
    coords2plot = coords.copy()
    coords2plot[:, [2, 0]] = coords2plot[:, [0, 2]]
    scene.add(Points(coords2plot, name=np.unique(group.probe_type).item(), colors="blackboard", radius=45))

# scene.add(PointsDensity(coordinates))

# scene.slice('sagittal')
# %%
# render
scene.content
scene.render(zoom=1.25)

# %%
# if ew=='k3d' - run these last two lines of code again and it'll render
if ew=='k3d':
    plt = Plotter()
    plt.show(*scene.renderables)

# %%

# %%
