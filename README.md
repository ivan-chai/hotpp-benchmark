<img src="./docs/logo.png" width="200">

# HoTPP: A Long-Horizon Event Sequence Prediction Benchmark
The benchmark is focused on the long-horizon prediction of event sequences, aka Marked Temporal Point Processes (MTPPs).

Each event is characterized by its timestamp, label and possible extra structured data.

# Installation
Sometimes the following parameters are necessary for a successful dependency installation:
```
CXX=<c++-compiler> CC=<gcc-compiler> pip install .
```

# Repository structure
The code is divided into the core library and dataset-specific scripts and configuration files.

Dataset-specific part is located in the experiments folder. Each subfolder includes data preparation scripts, model configuration files and README file. The data files and logs are usually stored in the same directory. All scripts must be invoked from the directory of the particular dataset. See individual README files for more details.

# Training and evaluation
To train the model use the following command:
```
python3 -m hotpp.train --config-dir configs --config-name <model>
```
A particular checkpoint can be evaluated with the following command:
```
python3 -m hotpp.evaluate --config-dir configs --config-name <model>
```
To run multiseed training and evaluation:
```
python3 -m hotpp.train_multiseed --config-dir configs --config-name <model>
```

# Library architecture
<p align="center">
<img src="./docs/hotpp-arch.png?raw=true" alt="Accuracy" width="75%"/>
</p>

HoTPP exploits high-level decomposition from PyTorch Lightning.

**DataModule.** All datasets are converted to a set of Parquet files. Each record in a Parquet file contains three main fields: *id*, *timestamps* and *labels*. The *id* field is a number representing identity associated with a sequence (user, client etc.). *Timestamps* are stored as an array of floating point numbers with a dataset-specific unit of measure. *Labels* is an array of integers representing a sequence of events types. Dataloader generates *PaddedBatch* object containing dictionary of padded sequences.

**Module.** *Module* implements high-level logic, specific for each group of methods. For example, there is a module for autoregressive models and another module for next-k approaches. *Module* incorporates a loss function, metric evaluator, and sequence encoder. Sequence encoder can produce discrete outputs, as in traditional RNNs, or it can be continous-time, like in NHP method.

**Trainer.** The *Trainer* object is typically should not be modified, except by a configuration file. *Trainer* uses *Module* and *DataModule* to train the model and evaluate metrics.

# Configuration files
HoTPP uses Hydra for configuration. The easiest way to make a new configuration file is to start from one in the experiments folder. Configuration file includes sections for logger, data module, module, and trainer. There are also some required top-level fields like `model_path` and `report`. It is also highly recommended to specify a random seed, by setting `seed_everything`.

# Hyperparameter tuning
Hyperparameters can be tuned by using [WandB Sweeps](https://docs.wandb.ai/guides/sweeps) mechanism. There are example configuration files for sweeps, like `experiments/age_pred/configs/sweep_next_item.yaml`. They can be used as follows:
```
wandb sweep ./configs/<sweep-configuration-file>
```
This command will generate a command for agent running, e.g.
```
wandb agent <sweep-id>
```
There is a special script in the library to analyze tuning results:
```
python3 -m hotpp.parse_wandb_hopt ./configs/<sweep-configuration-file> <sweep-id>
```

# Reproducibility
To achieve reproducible results it is highly recommended to use the provided Dockerfile. However, there can be minor differences depending on a particular GPU model.

# Tests
To run tests, use the following command:
```
pytest tests
```
