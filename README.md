# FlowMovie

[![License](https://img.shields.io/github/license/SFGLab/ChromMovie-flow)](LICENSE)
[![python](https://img.shields.io/badge/-Python_3.10+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)

## Reconstructing 3D Chromatin Dynamics Across the Cell Cycle with Multi-omic Flow Matching

<img width="590" height="490" alt="traj_FM_2D_contours" src="https://github.com/user-attachments/assets/60655147-a655-4b64-b272-6ad9fb50f915" />

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

## Repository Structure

```{text}
.
├── chrom_sizes/           # Files for the available chromosome size information
├── configs/               # Model, training, and experiment configuration files
├── data/                  # Data folder
├── models/                # Saved models (automatically created during training)
│   ├── interpolant/       # FlowMovie interpolant saved models
│   └── FM/                # FlowMovie flow matching saved models
├── notebooks/             # Jupyter notebooks for EDA, prototyping, and visualization
├── results/               # Figures and validation (automatically created after training)
├── scripts/               # Command-line tools for training, evaluation, and data prep
├── src/                   # Core package source code
│   ├── flow_matching/     # FlowMovie models, samplers, and loss functions
│   │   ├── interpolant/   # FlowMovie interpolant definition, scripts, and loss functions
│   │   └── flow_matching/ # FlowMovie flow matching definition, scripts, and loss functions
│   └── md/                # Molecular dynamics utilities
├── LICENSE                # Project license
├── pyproject.toml         # Package metadata, build setup
├── requirements.txt       # Package dependencies
└── README.md              # Project overview and usage guide
```

## Installing dependencies:

For optimal performance, please install the dependencies into a separate Python environment. For example using `pyenv`:

```{bash}
pyenv install -v 3.10.14
pyenv virtualenv 3.10.14 my_FlowMovie_env
pyenv activate my_FlowMovie_env
```

Install dependencies as specified in `requirements.txt`. To enable the scripts to import from `src` folder by running the following command in the main FlowMovie folder (where the `pyproject.toml` is located):

```{bash}
pip install -r requirements.txt
pip install -e .
```

## Running ChromMovie simulations:

For creating ChromMovie simulations. For the installation of ChromMovie please see the repository: [ChromMovie](https://github.com/SFGLab/ChromMovie).

Firstly, from `./src/md` folder prepare the data:

```{bash}
python preprocess_contacts.py -t "test02" -s "../../data/unique_cells_PATSKI_1k.txt" -c "../../data/final_dataset_downsampled_1k/filtered_ChIA-PET.bedpe" --n-traj 10 -n 3 -m 10 -ar 0.1
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
python FM_train.py -t "test02" -p "cycle" -s 100 -b 10 --viz-ot --ot-reg 0.1 -l 10 -ch 1500 -ct 200 -lr 1e-3 -fr 1e-3 -fm 'l2'
```

Create vizualization gifs for the trained FM velocity field. Trajectories will be iterated over time from `0` to the end time specified by the `-e` flag and number of plots specified by the `-n` flag.

```{bash}
python FM_viz_utils.py -t "test02" -p "cycle" -e 1.0 -n 10
python solvers.py -t "test02" -p "cycle" -r 2000000 --p 20 -d 300
```
