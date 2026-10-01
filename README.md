# FlowMovie

[![License](https://img.shields.io/github/license/SFGLab/ChromMovie-flow)](LICENSE)
[![python](https://img.shields.io/badge/-Python_3.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## Flow Matching chromatin structure generation based on ChromMovie simulations

Pipeline stages:

Preprocessing:
* Embed RNA-seq and ATAC-seq
* Prepare data for ChromMovie
* Run ChromMovie simulations
 
Flow Matching:
* Learn interpolant $\varphi$
* Compare the impact of phi on trajectories
* Train OT and final velocity field
* Compare and visualize resulting trajectories
 
Finally:
* Analyze

## Installing dependencies:

Enable the scripts to import from `src` folder by running the following command in the main FlowMovie folder (where the `pyproject.toml` is located):

```{bash}
pip install -e .
```

## Running ChromMovie simulations:

For creating ChromMovie simulations.

Firstly, from `./src/md` folder prepare the data:

```{bash}
python preprocess_contacts.py -t "test02" -s "../../data/unique_cells_PATSKI_1k.txt" -c "../../data/final_dataset_downsampled_1k/filtered_ChIA-PET.bedpe" --n-traj 10 -n 3 -m 10
```

Then prepare the configs:

```{bash}
python prepare_CM_configs.py -t "test02"
```

Finally, in the `./scripts` folder run the following (might need to specify the configs directory and ChromMovie folder):

```{bash}
./run_CM_simulations.sh ../configs/md/test02/ ../../ChromMovie
```

Alternatively:

```{bash}
for file in /home/baneckik/Documents/projects/2026_FM/FlowMovie/configs/md/test01/*.yaml; do python3 ../ChromMovie/ChromMovie.py -i "$file"; done
```

## Running interpolant training:

Move to `./src/flow_matching/interpolant` folder.

Preprocess the `.cif` file into `.pt`:

```{bash}
python preprocess_cif2pt.py -t "test02" -p "cycle"
```

Train neural interpolant for specified number of epochs (`-e` flag):

```{bash}
python interpolant_train.py -t "test02" -p "cycle" -e 10
```

Create diagnostics plots for a given trajectory ID (`-i` flag). Add `--plot-show` flag if figures are to be shown immediately.

```{bash}
python interpolant_diagnostics.py -t "test02" -p "cycle" -i 0
```

## Running FM training:

Move to `./src/flow_matching/flow_matching` folder.

Train neural Flow Matching velocity field:

```{bash}
python FM_train.py -t "test02" -p "cycle" -s 100 -b 10 --viz-ot --ot-reg 0.1
```

Create vizualization gif for the trained FM velocity field:

```{bash}
python FM_viz_utils.py -t "test02" -p "cycle"
```
