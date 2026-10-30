# FlowMovie Modules

This file summarizes the building blocks of the Flow Matching part of the FlowMovie pipeline as well as provides instructions on how to use it.

## FM_model.py

**Main file containing the definition of the Flow Matching network.**

At the present moment the FM is implemented as a MLP with a number of fully connected layers. The time is embedded into the network via Sinusoidal Time Embedding. 

## FM_train.py

**Main file to use for the training of the FM model.**

The training is performed using a $\varphi$ interpolant and data specified by the `test_name` and `phase_trans` parameters. 

The data is splitted via 80/20 train/test split. The trained model is saved into `models/FM` folder. The resulting loss functions and other diagnostics are saved into the `results/FM` folder.

To train the model:

```{bash}
python FM_train.py
```

## OT.py

**Main file implementing the Optimal Transport (OT) plan used in the training of the FM model.**

The file implements the Sinkhorn algorithm for the calculation of the OT. Main function used by the `FM_train.py` is the `get_OT_x1()` which goal is to alter the `x1` tensor in accordance with the OT.

## FM_viz_utils.py

**File containing the visualization and diagnostics functions for the FM part of the FlowMovie pipeline.**

The main diagnostics plots include the plotting of the losses during FM training and trajectory gif. The resulting figures are saved into the `results/FM` folder.

To create vizualizations of the FM training:

```{bash}
python FM_viz_utils.py
```

## Solvers.py

**File containing the implementation of the Euler and RK4 iteration throught the FM velocity field**

The velocity field model trained with `FM_train.py` and specified by the `test_name` and `phase_trans` parameters is used. The file additionally contains the implementation of the 3D structure trajectory visualization.

