# IMPORTANT
- save trial and trialtm to the nwbfile itself (maybe, it's not as slow rn as I thought it would be)
- stimLoc **
    - photostim_start_times.control -> ogen_sites    -> tells you the hemisphere
- stimEpoch
    - this is just using photostim_start and end times
- lick raster
    - augment color based on lick dir and cond, not sure how yet tho
- heatmap time axes are so weird, need to write custom function to plot them nicely
- make an ALM .obj mesh
- tongue length in getKinematics

- combine nwbfile.units.brain_region's left/right label with allenccf region
