# INSTRUCTIONS
## ON WINDOWS
- `conda create --name brainrender python=3.8 -y`
- `conda activate brainrender`
- `pip install brainrender`
- `pip install ipyvtklink` 
- find where brainrender uses `np.float` and change to `np.float64`. It's just in one function in `scene.points` or something like that
- use k3d

## ON MAC
- `conda create --name brainrender python=3.6 -y`
- `conda activate brainrender`
- `pip install brainrender` 
- `pip install ipyvtklink` 
- use k3d or no embedded window

## NOTES
- I added an `alpha` input to brainrender.Scene so that I can control the alpha of the `root` (the outline of the brain)
    - see `/Users/munib/anaconda3/envs/brainrender/lib/python3.6/site-packages/brainrender/scene.py`