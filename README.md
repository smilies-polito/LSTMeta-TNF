## Experimental setup

Follow these steps to setup for reproducing the experiments provided in ....
1) Install `Singularity` from https://docs.sylabs.io/guides/3.0/user-guide/installation.html:
	* Install `Singularity` release 3.10.2, with `Go` version 1.18.4
	* Suggestion: follow instructions provided in _Download and install singularity from a release_ section after installing `Go`
	* Install dependencies from: https://docs.sylabs.io/guides/main/admin-guide/installation.html
2) Clone the lstmeta-tnf repository in your home folder
```
git clone https://github.com/smilies-polito/lstmeta-tnf.git #change the link
```
3) Move to the lstmeta-tnf source subfolder, and build the singularity container with 
```
cd  lstmeta-tnf/source
sudo singularity build lstmeta-tnf.sif lstmeta-tnf.def
```
or
```
cd lstmeta-tnf/source
singularity build --fakeroot lstmeta-tnf.sif lstmeta-tnf.def
```

## Reproducing the analysis interactively within the lstmeta-tnf Singularity container

To run analyses manually launch the lstmeta-tnf Singularity container, move to `/source`, and launch the scripts as follows.

First of all, launch the lstmeta-tnf Singularity container
```
cd source
singularity shell --nv --bind /path/to/your/data/folder:/mnt lstmeta-tnf.sif
```
This will run a shell within the container, and the following prompt should appear:
```
Singularity>
```
Using this prompt, follow the steps below. 

### Dataset creation

1) move to simulations folder and execute the data_extraction_definitive.py script
```
cd ../simulations
python data_extraction_definitive.py
```
the data_extraction_definitive.py execute the 7288 different simulations and saves the input parameters and the cell states of each simulation respectively in the /path/to/your/data/folder/data/input_parameters and /path/to/your/data/folder/data/cell_data folders


## Reproducing the analysis running the Singularity container

To reproduce the analysis from this paper, run the singularity container lstmeta-tnf.sif

1) Move to the lstmeta-tnf/source folder and run the lstmeta-tnf.sif file
```
cd lstmeta-tnf/source
singularity run --nv --bind /path/to/your/data/folder:/mnt lstmeta-tnf.sif 
```


