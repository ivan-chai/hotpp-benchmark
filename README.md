<h3 align="center">
  <img src="./docs/logo.png" width="200">
</h3>

<h1 align="center"> HoTPP: A Long-Horizon Event Sequence Prediction Benchmark</h1>
<h4 align="center">
    <p>
        <a href="#Installation">Installation</a> |
        <a href="#Training and evaluation">Usage</a> |
        <a href="#Evaluation results">Results</a> |
        <a href="#Citation">Citing</a>
    <p>
</h4>


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

# Evaluation results
### Transactions
| Method              | Acc.      | MAE       | OTD  Val / Test     | T-mAP  Val / Test   |
|:--------------------|:----------|:----------|:--------------------|:--------------------|
| MostPopular         | 32.78     | 0.763     | 7.41 / 7.42         | 1.08 / 1.01         |
| Last 5              | 19.61     | 0.948     | 7.53 / 7.57         | 1.65 / 1.65         |
| MAE-CE              | **38.19** | **0.635** | **6.79** / **6.81** | 6.08 / 5.98         |
| MAE-CE-K            | 37.55     | 0.640     | 7.15 / 7.16         | 4.28 / 4.18         |
| RMTPP               | 38.17     | 0.696     | 6.82 / 6.83         | 7.37 / 6.85         |
| RMTPP-K             | 37.56     | 0.697     | 7.09 / 7.10         | 5.60 / 5.40         |
| NHP                 | 35.43     | 0.707     | 7.01 / 7.02         | 5.57 / 5.58         |
| ODE                 | 35.59     | 0.707     | 7.00 / 7.01         | 5.48 / 5.53         |
| HYPRO               | 34.26     | 0.770     | 7.11 / 7.12         | **7.84** / **7.12** |

### MIMIC-IV
| Method              | Acc.      | MAE       | OTD  Val / Test       | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:----------------------|:----------------------|
| MostPopular         | 1.79      | 50.20     | 19.89 / 19.88         | 0.73 / 0.75           |
| Last 5              | 0.88      | 48.71     | 19.77 / 19.74         | 2.31 / 2.38           |
| MAE-CE              | **57.39** | **26.93** | **11.69** / **11.70** | **23.47** / **23.60** |
| MAE-CE-K            | 56.70     | 26.99     | 13.93 / 13.89         | 21.27 / 21.05         |
| RMTPP               | 57.12     | 27.83     | 13.77 / 13.80         | 21.53 / 21.44         |
| RMTPP-K             | 56.27     | 27.60     | 14.92 / 14.88         | 19.49 / 19.55         |
| NHP                 | 24.30     | 29.72     | 18.51 / 18.51         | 6.15 / 6.14           |
| ODE                 | 44.02     | 27.29     | 14.70 / 14.67         | 14.60 / 14.62         |
| HYPRO               | 56.91     | 27.94     | 14.94 / 14.96         | 12.50 / 12.39         |

### Retweet
| Method              | Acc.      | MAE       | OTD  Val / Test   | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:------------------|:----------------------|
| MostPopular         | 58.50     | 26.27     | 174.9 / 173.5     | 26.51 / 25.87         |
| Last 10             | 50.29     | 30.20     | 165.4 / 163.7     | 28.62 / 28.75         |
| MAE-CE              | 59.95     | **24.31** | 173.3 / 172.7     | 33.49 / 31.75         |
| MAE-CE-K            | 59.55     | 25.99     | 169.1 / 168.5     | 36.64 / 34.27         |
| RMTPP               | 60.07     | 25.52     | 167.7 / 166.6     | 48.22 / 45.41         |
| RMTPP-K             | 59.99     | 25.85     | 165.1 / 164.3     | 48.66 / 45.76         |
| NHP                 | **60.09** | 25.50     | 167.0 / 165.7     | 49.57 / 46.91         |
| ODE                 | 59.94     | 25.47     | 166.6 / 165.3     | 48.96 / 45.96         |
| HYPRO               | 59.87     | 25.85     | 172.8 / 171.9     | 53.47 / 50.67         |
| DeTPP               | 58.02     | 27.08     | **147.7** / **145.6** | **58.38** / **55.22** |

### Amazon
| Method              | Acc.      | MAE       | OTD  Val / Test     | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:--------------------|:----------------------|
| MostPopular         | 33.46     | 0.304     | 7.20 / 7.18         | 9.34 / 9.02           |
| Last 5              | 24.23     | 0.321     | 6.70 / 6.67         | 9.73 / 9.21           |
| MAE-CE              | 35.73     | **0.242** | 6.62 / 6.55         | 21.99 / 22.60         |
| MAE-CE-K            | 35.11     | 0.246     | 6.74 / 6.70         | 21.84 / 22.35         |
| RMTPP               | **35.75** | 0.294     | 6.67 / 6.62         | 20.06 / 20.42         |
| RMTPP-K             | 35.06     | 0.300     | 6.94 / 6.89         | 17.63 / 17.89         |
| NHP                 | 11.00     | 0.449     | 9.05 / 9.04         | 25.05 / 24.91         |
| ODE                 | 7.61      | 0.492     | 9.47 / 9.48         | 22.50 / 21.88         |
| HYPRO               | 35.69     | 0.295     | 6.72 / 6.67         | 21.03 / 21.31         |
| DeTPP               | 34.08     | 0.291     | **6.34** / **6.30** | **36.82** / **37.13** |

### StackOverflow
| Method              | Acc.      | MAE       | OTD  Val / Test       | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:----------------------|:----------------------|
| MostPopular         | 42.90     | 0.744     | 13.56 / 13.77         | 6.33 / 5.86           |
| Last 10             | 26.42     | 0.934     | 14.84 / 14.90         | 8.67 / 6.72           |
| MAE-CE              | 45.41     | **0.641** | 13.58 / 13.65         | 8.85 / 8.46           |
| MAE-CE-K            | 44.85     | 0.644     | 13.41 / 13.52         | 12.14 / 11.17         |
| RMTPP               | **45.43** | 0.700     | 12.97 / 13.18         | 13.39 / 12.99         |
| RMTPP-K             | 44.89     | 0.687     | 12.91 / 13.13         | 14.65 / 14.12         |
| NHP                 | 44.54     | 0.716     | 13.02 / 13.25         | 12.62 / 11.93         |
| ODE                 | 44.37     | 0.713     | 13.04 / 13.28         | 11.25 / 10.50         |
| HYPRO               | 45.18     | 0.714     | 13.04 / 13.30         | 15.74 / 14.92         |
| DeTPP               | 44.78     | 0.675     | **12.63** / **12.77** | **20.19** / **19.69** |

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
