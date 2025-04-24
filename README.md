<h1> <img align="left" src="./docs/logo.png" width="15%"> HoTPP: A Long-Horizon Event Sequence Prediction Benchmark </h1>

<div align="center">

  <a href="">[![PyPI version](https://badge.fury.io/py/hotpp-benchmark.svg)](https://badge.fury.io/py/hotpp-benchmark)</a>
  <a href="">[![Build Status](https://github.com/ivan-chai/hotpp-benchmark/actions/workflows/ci-tests.yml/badge.svg)](https://github.com/ivan-chai/hotpp-benchmark/actions)</a>
  <a href="">[![Downloads](https://static.pepy.tech/badge/hotpp-benchmark)](https://pepy.tech/project/hotpp-benchmark)</a>
  <a href="">[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)</a>

</div>

<h4 align="center">
    <p>
        <a href="#Installation">Installation</a> |
        <a href="#Training-and-evaluation">Usage</a> |
        <a href="#Evaluation-results">Results</a> |
        <a href="#Library-architecture">Extension</a> |
        <a href="https://arxiv.org/pdf/2406.14341">Paper HoTPP</a> |
        <a href="https://arxiv.org/pdf/2408.13131">Paper DeTPP</a> |
        <a href="#Citation">Citing</a>
    <p>
</h4>


The HoTPP benchmark focuses on the long-horizon prediction of event sequences. Each event is characterized by its timestamp, label, and possible additional structured data. Sequences of this type are also known as Marked Temporal Point Processes (Marked TPP, MTPP).

# Features
* Next event prediction
* Long-horizon prediction
* Downstream classification
* Working with heterogeneous input and output fields (general event sequences)
* Distributed training
* RNN and Transformer models (including HuggingFace causal models)
* Discrete and continuous-time models
* Improved TPP thinning algorithm
* Optimized training and inference (ODE, cont. time models, multi-point generation with RNN)

# Implemented Methods
The list of implemented papers:

| Year | Name      | Paper                                                                                                      | Source                           |
|------|-----------|------------------------------------------------------------------------------------------------------------|----------------------------------|
| 2025 | Diffusion | Non-autoregressive diffusion-based temporal point processes for continuous-time long-term event prediction | Expert Systems with Applications |
| 2024 | DeTPP     | DeTPP: Leveraging Object Detection for Robust Long-Horizon Event Prediction                                | arXiv                            |
| 2022 | AttNHP    | Transformer embeddings of irregularly spaced events and their participants                                 | ICLR 2022                        |
| 2022 | HYPRO     | Hypro: A hybridly normalized probabilistic model for long-horizon prediction of event sequences            | NeurIPS 2022                     |
| 2020 | IFTPP     | Intensity-free learning of temporal point processes                                                        | ICLR 2020                        |
| 2019 | ODE       | Latent ordinary differential equations for irregularly-sampled time series                                 | NeurIPS 2019                     |
| 2017 | NHP       | The neural hawkes process: A neurally self-modulating multivariate point process                           | NeurIPS 2017                     |
| 2016 | RMTPP     | Recurrent marked temporal point processes: Embedding event history to vector                               | SIGKDD 2016                      |

Other methods:
* Simple baselines (MostPopular, Last-K)
* Next-K extensions of IFTPP and RMTPP.
* Transformer variants of RMTPP and IFTPP.

# Installation
Install via PyPI:
```sh
pip install hotpp-benchmark
```

To install downstream evaluation tools:
```sh
pip install 'hotpp-benchmark[downstream]'
```

Sometimes the following parameters are necessary for [successful dependency installation](https://github.com/ivan-chai/torch-linear-assignment?tab=readme-ov-file#install):
```sh
CXX=<c++-compiler> CC=<gcc-compiler> pip install hotpp-benchmark
```

# Training and evaluation
The code is divided into the core library and dataset-specific scripts and configuration files.

The dataset-specific part is located in the `experiments` folder. Each subfolder includes data preparation scripts, model configuration files, and a README file. Data files and logs are usually stored in the same directory. All scripts must be executed from the directory of the specific dataset. Refer to the individual README files for more details.

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

To run multi-GPU training on 2 GPUs:
```sh
mpirun -np 2 python3 -m hotpp.train --config-dir configs --config-name <model> ++trainer.devices=2 ++trainer.strategy=ddp
```

# Evaluation results
All evaluation results can be found in the experiments folder.

[See tables.](docs/RESULTS.md)

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

# Known issues
If downstream evaluation hangs during LightGBM or CatBoost training, try setting the following environment variable:
```sh
OMP_NUM_THREADS=1 python3 -m hotpp.evaluate --config-dir configs --config-name <model>
```

# Citation
If you use [HoTPP](https://arxiv.org/pdf/2406.14341) in your project, please cite the following paper:
```
@article{karpukhin2024hotppbenchmark,
  title={HoTPP Benchmark: Are We Good at the Long Horizon Events Forecasting?},
  author={Karpukhin, Ivan and Shipilov, Foma and Savchenko, Andrey},
  journal={arXiv preprint arXiv:2406.14341},
  year={2024},
  url ={https://arxiv.org/abs/2406.14341}
}
```

If you incorporate ideas from [DeTPP](https://arxiv.org/pdf/2408.13131), use it for comparison, or reference it in a review, please cite the following paper:
```
@article{karpukhin2024detpp,
  title={DeTPP: Leveraging Object Detection for Robust Long-Horizon Event Prediction},
  author={Karpukhin, Ivan and Savchenko, Andrey},
  journal={arXiv preprint arXiv:2408.13131},
  year={2024},
  url ={https://arxiv.org/abs/2408.13131}
}
```
