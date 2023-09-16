# %%
# # INSTRUCTIONS
# ON WINDOWS
# conda create --name brainrender python=3.8 -y
# conda activate brainrender
# pip install brainrender 
# pip install ipyvtklink 
# find where brainrender uses np.float and change to np.float64. It's just in one function in scene.points or something like that
# use k3d
# USE VS CODE

# ON MAC
# conda create --name brainrender python=3.6 -y
# conda activate brainrender
# pip install brainrender 
# pip install ipyvtklink 
# use no embedded window
# use jupyter notebook
# can also use vs code and k3d actually

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

sub = '484676' # subject/animal id
date = '20210420' # session date

df = utils.loadCoordinates(dataDir,sub,date)

# %%

ew = 'k3d' # render in interactive plot in vscode
# ew = None  # render in separate window outside of jupyer notebook
embedWindow(ew)

scene = Scene(title="Labelled cells",atlas_name="allen_mouse_100um", inset=False)
if ew == 'k3d': 
    scene.jupyter = True

mos = scene.add_brain_region("MOs",color='yellow', alpha=0.15)
irn = scene.add_brain_region("IRN", alpha=0.15)

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

# render
scene.content
scene.render(zoom=1.25)

# %%
# if ew=='k3d' - run these last two lines of code again and it'll render
plt = Plotter()
plt.show(*scene.renderables)

# %%

# %%
