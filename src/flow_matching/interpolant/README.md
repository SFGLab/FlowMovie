# $\varphi$ Interpolant Modules

This file summarizes the building blocks of the $\varphi$ interpolant part of the FlowMovie pipeline as well as provides instructions on how to use it.

## interpolant.py

**Main file containing the definition of the $\varphi$ interpolant network.**

Currently the interpoland model is implemented as a Multi-Head Transformer with Sinusoidal Positional Embedding used for time dimension. 

## interpolant_train.py

**Main function for $\varphi$ interpolant training.**

The training takes the data specified by the `test_name` and `phase_trans` parameters and saves the trained model in the `models/interpolant` folder. 

To train the model:

```{bash}
python interpolant_train.py
```

## interpolant_diagnostics.py

**File containing the plotting and diagnostics for the interpolant training.**

The file contains implementation of a number of diagnostics plots including consequtive distance and angle distributions as well as RMSD measurements and 3D structure visualizations. The plots are saved in the `results/interpolant` folder.

To run diagnostics plot for the interpolant model:

```{bash}
python interpolant_diagnostics.py
```

## interpolant_DataLoader.py

**File containg the implementation of the data loading module used for the interpolant training.**

The implementation allows to specify the batch size before loading the data for training. The 3D structure data need to be firstly preprocessed into the `.pt` format (for example via `preprocess_cif2pt.py`) function described below. Example use of the data loading is provided.

## preprocess_cif2pt.py

**This file contains the implementation of the preprocessing of the ChromMovie trajectories.**

The script takes both the ChromMovie input and output directories (see `README.md` in the `src/md` folder for details). The processed output is saved and stored as a single `.pt` file.

To run the preprocessing of the ChroMovie simulation:

```{bash}
python preprocess_cif2pt.py
```

## pt_file_check.py

Simple script for checking whether the generated `.pt` file was saved correctly.

