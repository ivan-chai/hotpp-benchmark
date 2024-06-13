<img src="./docs/logo.png" width="200">

# HoTPP: A Long-Horizon Event Sequence Prediction Benchmark
The HoTPP benchmark focuses on the long-horizon prediction of event sequences. Each event is characterized by its timestamp, label, and possible additional structured data. Sequences of this type are also known as Marked Temporal Point Processes (MTPPs).

# Installation
Sometimes the following parameters are necessary for successful dependency installation:
```sh
CXX=<c++-compiler> CC=<gcc-compiler> pip install .
```

# Repository structure
The code is divided into the core library and dataset-specific scripts and configuration files.

The dataset-specific part is located in the `experiments` folder. Each subfolder includes data preparation scripts, model configuration files, and a README file. Data files and logs are usually stored in the same directory. All scripts must be executed from the directory of the specific dataset. Refer to the individual README files for more details.

# Training and evaluation
To train the model, use the following command:
```sh
python3 -m hotpp.train --config-dir configs --config-name <model>
```

To evaluate a specific checkpoint, use the following command:
```sh
python3 -m hotpp.evaluate --config-dir configs --config-name <model>
```

To run multiseed training and evaluation:
```sh
python3 -m hotpp.train_multiseed --config-dir configs --config-name <model>
```

# Library architecture
<p align="center">
<img src="./docs/hotpp-arch.png?raw=true" alt="Accuracy" width="75%"/>
</p>

HoTPP leverages high-level decomposition from [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/index.html).

**DataModule.** All datasets are converted to a set of Parquet files. Each record in a Parquet file contains three main fields: *id*, *timestamps*, and *labels*. The *id* field represents the identity associated with a sequence (user, client, etc.). *Timestamps* are stored as an array of floating point numbers with a dataset-specific unit of measure. *Labels* is an array of integers representing a sequence of event types. The dataloader generates a *PaddedBatch* object containing a dictionary of padded sequences.

**Module.** The *Module* implements high-level logic specific to each group of methods. For example, there is a module for autoregressive models and another for next-k approaches. The *Module* incorporates a loss function, metric evaluator, and sequence encoder. The sequence encoder can produce discrete outputs, as in traditional RNNs, or continuous-time outputs, as in the NHP method.

**Trainer:** The *Trainer* object should typically not be modified, except through a configuration file. The *Trainer* uses the *Module* and *DataModule* to train the model and evaluate metrics.

# Configuration files
HoTPP uses [Hydra](https://hydra.cc/) for configuration. The easiest way to create a new configuration file is to start from one in the `experiments` folder. The configuration file includes sections for the logger, data module, module, and trainer. There are also some required top-level fields like `model_path` and `report`. It is highly recommended to specify a random seed by setting `seed_everything`.

# Hyperparameter tuning
Hyperparameters can be tuned by [WandB Sweeps](https://docs.wandb.ai/guides/sweeps). Example configuration files for sweeps, such as `experiments/amazon/configs/sweep_next_item.yaml`, can be used as follows:
```sh
wandb sweep ./configs/<sweep-configuration-file>
```

The above command will generate a command for running the agent, e.g.:
```sh
wandb agent <sweep-id>
```

There is a special script in the library to analyze tuning results:
```sh
python3 -m hotpp.parse_wandb_hopt ./configs/<sweep-configuration-file> <sweep-id>
```

# Reproducibility
To achieve reproducible results, it is highly recommended to use the provided Dockerfile. However, there may be minor differences depending on the specific GPU model.

The reference evaluation results are stored in the `results` subfolder within each dataset directory in the `experiments` folder.

# Tests
To run tests, use the following command:
```sh
pytest tests
```
