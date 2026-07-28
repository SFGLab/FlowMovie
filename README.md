# FlowMovie

[![License](https://img.shields.io/github/license/SFGLab/ChromMovie-flow)](LICENSE)

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

### Running ChromMovie for the ready config .yaml files

For creating ChromMovie simulations.

Firstly, from `./src/md` folder prepare the data:

```{bash}
python preprocess_contacts.py
```

Then prepare the configs:

```{bash}
python prepare_CM_configs.py
```

Finally in the `./scripts` folder run the following (might need to specify the configs directory and ChromMovie folder):

```{bash}
./run_CM_simulations.sh
```

Alternatively:

```{bash}
for file in /home/baneckik/Documents/projects/2026_FM/FlowMovie/configs/md/test01/*.yaml; do python3 ../ChromMovie/ChromMovie.py -i "$file"; done
```

