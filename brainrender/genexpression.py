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
from brainrender import settings
from brainrender.atlas_specific import GeneExpressionAPI

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

settings.SHOW_AXES = False
scene = Scene(inset=False)

gene = "Gpr161"
geapi = GeneExpressionAPI()

expids = geapi.get_gene_experiments(gene)
data = geapi.get_gene_data(gene, expids[1])

gene_actor = geapi.griddata_to_volume(data, min_quantile=99, cmap="coolwarm")
act = scene.add(gene_actor)

# ca1 = scene.add_brain_region("CA1", alpha=0.2, color="skyblue")
# ca3 = scene.add_brain_region("CA3", alpha=0.5, color="salmon")
ca3 = scene.add_brain_region("grey", alpha=0.5, color="salmon")


scene.add_silhouette(act)

scene.render(zoom=1.6)

scene.render(zoom=1.25)

# %%
# if ew=='k3d' - run these last two lines of code again and it'll render
plt = Plotter()
plt.show(*scene.renderables)

# %%

# %%
