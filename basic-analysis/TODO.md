# IMPORTANT
- using units_df.depth, units_df.xpos/ypos and units_df.probe_type, and units_df.ml/ap_coordinates, find which region each unit was in **
    - some sessions are recording from multiple regions along the DV axis
    - can do this by 'registering' to allen CCF coordinates
    - this might not be necessary since units were registered to the CCF, but not sure if that is what's provided in the dataset
    - for medulla sessions, look at xpos/ypos on probe. should all be on bank 1
- stimLoc **
    - Potential stim locations are in nwbfile.ogen_sites. there's 3 entries in all the sessions I checked. The first entry is the left hemisphere, second entry is the right hemisphere of that same region, and then the third entry is both of those concatenated which is probably bilateral then. buuut I don't see a reference to those entries anywhere, so don't know how to tell if a trial was left/right/bilateral stim
-stimEpoch
- where each unit actually was recorded

- fix plotting raster
    - should use trial start and end times
    - see lickRaster()
    - getting psths might also need to take into account tstart and tend *************
- add trialtm and trial 
    - fix plotPSTH raster after you do this
    - needs to take into account trial start and end times
- give this to the lab to use
- GET KINEMATICS (make it like our obj.traj)
- lick raster
    - augment color based on lick dir and cond, not sure how yet tho
- heatmap axes





