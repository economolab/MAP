# code to analyze data from map-ephys project
### (Susu Chen, Thinh Nguyen, Nuo Li, Karel Svoboda)

## [LINK TO DATASET](https://dandiarchive.org/dandiset/000363?search=susu+chen&pos=1)

## DOWNLOAD INSTRUCTIONS

- you can download from the link above directly
- alternatively, after pip installing `dandi` you can run the following command:
	- `dandi download DANDI:000363 -o path/to/output/folder`

## DANDI - https://www.dandiarchive.org/handbook/12_download/

#### DOWNLOAD DATA 
- `dandi download DANDI:datasetID`

- to specify an output path `dandi download DANDI:000363 -o output/path/here`           (NOTE: downloads all data stored in dandi repository)
	
- use a url to the specific dataset you want to download if you don't want to download entire dataset

- you can also point to the data via a url rather than downloading. See here: https://www.dandiarchive.org/handbook/12_download/#download-a-specific-file-from-a-dandiset
	
	
## Other comments
- recommend making a new CONDA ENVIRONMENT and pip install `dandi matplotlib jupyter numpy pandas scikit-learn` and `torch` if you have a GPU.
	- look up current pytorch pip install instructions since it changes often and is very specific to a given setup. 

- nwbfile.acquisition['BehavioralEvents'][eventName].timestamps[:]
	- use timestamps field rather than data here. 
	
- can use hdfview app to get overview of fields, but it's easier to look at stuff in python. couldn't get nwbviewer to install properly

Comments:
1. to get widgets to work in vs code, run:
	`jupyter nbextension enable --py --sys-prefix widgetsnbextension`
2. to have interactive plots:
	- `pip install ipympl`
	- in your code: `%matplotlib widget`
3. I regularly use jupyter notebooks when coding in python, the biggest pain is having to restart the kernel when you make updates to a module. I found this workaround from stackoverflow however. Just re-run the code below anytime you update a module and the updates are reflected in the current workspace/kernel without needing to restart. 
- https://stackoverflow.com/questions/66828031/do-i-always-have-to-restart-my-kernel-in-jupyter-lab-when-code-in-a-local-module
- `_ = importlib.reload(sys.modules['nameOfModule'])`
4. if going to re-upload data to dandi, should include a field called trialtm and trial to make binning data and getting PSTHs ~100x faster