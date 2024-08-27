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

3) Move to the lstmeta-tnf source subfolder, and build the Singularity container with 
```bash
cd  lstmeta-tnf/source
sudo singularity build lstmeta-tnf.sif lstmeta-tnf.def
```
or using fake root privileges
```bash
cd lstmeta-tnf/source
singularity build --fakeroot lstmeta-tnf.sif lstmeta-tnf.def
```

## Reproducing the analysis interactively within the lstmeta-tnf Singularity container

To run analyses manually launch the lstmeta-tnf Singularity container. Move to the `source` folder, and launch the scripts as follows.

First of all, launch the lstmeta-tnf Singularity container
```bash
cd source
singularity shell --fakeroot --nv --bind /path/to/your/data/folder:/mnt lstmeta-tnf.sif
```
This will run a shell within the container, and the following prompt should appear:
```bash
Singularity>
```
Then open a bash shell by running
```bash
bash
```
Now follow the steps below. 

#### Dataset creation

Move to the `simulations` folder and execute the `data_extraction_definitive.py` script
```bash
cd lstmeta-tnf/simulations
python data_extraction_definitive.py
```
the `data_extraction_definitive.py` execute the 7,288 different simulations and saves the input parameters and the cell states of each simulation respectively in the `/path/to/your/data/folder/data/input_parameters` and `/path/to/your/data/folder/data/cell_data` folders.

#### Dataset analysis and CSVs creation

Move to the `metamodel` folder and run the `data_exploration.ipynb` notebook. From the terminal
```bash
cd ../metamodel
jupyter nbconvert --execute --to notebook --inplace data_exploration.ipynb
```

#### Metamodel training

You can train a new metamodel LSTM by running the `rec_model_train.py` script. You can also specify the `tumor_radius` (i.e. 50, 100, 275, 400), the highest latent dimension `latent_dimension` (default 1000) of the model, the window length `window_size` (default 24) of the sequences, the number of latent layers `n_layers_in` (default 2) in the encoder and decoder, the linear dropout `dropout_lin` (default 0), and the recurrent dropout `dropout_rec` (default 0.2).
```bash
python rec_model_train.py --tumor_radius 50 --latent_dim 1000 --window_size 24 --n_layers_in 2 --dropout_lin 0 --dropout_rec 0.2
```

The results will be saved in a new `metamodel/experiments` folder.


## Reproducing the analysis running the Singularity container

To reproduce the analysis from this paper, run the singularity container lstmeta-tnf.sif

Move to the `source` folder and run the `lstmeta-tnf.sif` file
```bash
cd source
singularity run --fakeroot --nv --bind /path/to/your/data/folder:/mnt lstmeta-tnf.sif 
```


