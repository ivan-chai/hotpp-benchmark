<h1> <img align="left" src="./docs/logo.png" width="15%"> HoTPP: A Long-Horizon Event Sequence Prediction Benchmark </h1>

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


The HoTPP benchmark focuses on the long-horizon prediction of event sequences. Each event is characterized by its timestamp, label, and possible additional structured data. Sequences of this type are also known as Marked Temporal Point Processes (MTPPs).

# Installation
Sometimes the following parameters are necessary for successful dependency installation:
```sh
CXX=<c++-compiler> CC=<gcc-compiler> pip install .
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

# Evaluation results
### Transactions
| Method              | Acc.      | mAP      | MAE       | OTD  Val / Test     | T-mAP  Val / Test   |
|:--------------------|:----------|:---------|:----------|:--------------------|:--------------------|
| MostPopular         | 32.78     | 0.86     | 0.752     | 7.37 / 7.38         | 1.06 / 0.99         |
| RecentHistory       | 19.60     | 0.87     | 0.924     | 7.40 / 7.44         | 2.46 / 2.49         |
| MAE-CE              | 34.08     | 3.47     | 0.693     | 6.88 / 6.90         | 5.82 / 5.88         |
| MAE-CE-K            | 33.69     | 3.25     | 0.698     | 7.18 / 7.19         | 4.42 / 4.43         |
| RMTPP               | 34.15     | 3.47     | 0.749     | 6.86 / 6.88         | 7.08 / 6.69         |
| RMTPP-K             | 33.63     | 3.24     | 0.749     | 7.10 / 7.11         | 5.82 / 5.52         |
| NHP                 | 35.43     | 3.41     | 0.696     | 6.97 / 6.98         | 5.59 / 5.61         |
| ODE                 | 35.60     | 3.34     | 0.695     | 6.96 / 6.97         | 5.53 / 5.52         |
| HYPRO               | 34.26     | 3.46     | 0.758     | 7.04 / 7.05         | 7.79 / 7.05         |
| DeTPP               | 35.04     | 3.85     | 0.688     | **6.65** / **6.66** | 9.18 / 9.17         |
| DeTPP-Hybrid        | **38.29** | **4.67** | **0.638** | 6.68 / 6.70         | **9.59** / **9.26** |

### MIMIC-IV
| Method              | Acc.      | mAP       | MAE      | OTD  Val / Test       | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:---------|:----------------------|:----------------------|
| MostPopular         | 4.77      | 2.75      | 14.52    | 19.82 / 19.82         | 0.55 / 0.54           |
| RecentHistory       | 1.02      | 2.63      | 5.34     | 19.75 / 19.73         | 2.41 / 2.49           |
| MAE-CE              | 58.59     | **47.32** | 3.00     | **11.51** / **11.53** | 21.93 / 21.67         |
| MAE-CE-K            | 57.91     | 44.60     | 3.07     | 13.17 / 13.18         | 22.46 / 22.30         |
| RMTPP               | 58.33     | 46.24     | 3.89     | 13.64 / 13.71         | 21.49 / 21.08         |
| RMTPP-K             | 57.48     | 43.47     | 3.62     | 14.68 / 14.72         | 20.70 / 20.39         |
| NHP                 | 24.97     | 11.12     | 6.53     | 18.59 / 18.60         | 7.26 / 7.32           |
| ODE                 | 43.21     | 25.34     | 2.93     | 14.71 / 14.74         | 15.41 / 15.18         |
| DeTPP               | 28.62     | 25.44     | **2.74** | 12.86 / 12.85         | **30.92** / **30.63** |
| DeTPP-Hybrid        | **58.66** | 46.56     | 3.01     | 12.94 / 12.95         | 30.73 / 30.35         |

### Retweet
| Method              | Acc.      | mAP       | MAE       | OTD  Val / Test       | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:----------|:----------------------|:----------------------|
| MostPopular         | 58.50     | 39.85     | 18.82     | 174.9 / 173.5         | 25.15 / 23.91         |
| RecentHistory       | 50.29     | 35.73     | 21.87     | 152.3 / 150.3         | 29.12 / 29.24         |
| MAE-CE              | 59.95     | 46.53     | 18.27     | 173.3 / 172.7         | 34.90 / 31.75         |
| MAE-CE-K            | 59.55     | 45.09     | **18.21** | 168.6 / 167.9         | 37.11 / 34.73         |
| RMTPP               | 60.07     | 46.81     | 18.45     | 167.6 / 166.7         | 47.86 / 44.74         |
| RMTPP-K             | 59.99     | 46.34     | 18.33     | 164.7 / 163.9         | 49.07 / 46.16         |
| NHP                 | **60.08** | **46.83** | 18.42     | 167.0 / 165.8         | 48.31 / 45.07         |
| ODE                 | 59.95     | 46.65     | 18.38     | 166.5 / 165.3         | 48.70 / 44.81         |
| HYPRO               | 59.87     | 46.69     | 18.75     | 171.4 / 170.7         | 49.90 / 46.99         |
| DeTPP               | 59.46     | 45.82     | 18.34     | 137.9 / 134.4         | 60.96 / 57.37         |
| DeTPP-Hybrid        | 60.04     | 46.76     | 18.35     | **136.4** / **132.9** | **61.47** / **57.93** |

### Amazon
| Method              | Acc.      | mAP       | MAE       | OTD  Val / Test     | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:----------|:--------------------|:----------------------|
| MostPopular         | 33.46     | 9.58      | 0.304     | 7.20 / 7.18         | 8.59 / 8.31           |
| RecentHistory       | 24.23     | 8.15      | 0.321     | 6.43 / 6.41         | 9.73 / 9.21           |
| MAE-CE              | 35.73     | 17.14     | 0.242     | 6.58 / 6.52         | 21.94 / 22.56         |
| MAE-CE-K            | 35.11     | 16.48     | 0.246     | 6.72 / 6.68         | 22.06 / 22.57         |
| RMTPP               | 35.76     | 17.21     | 0.294     | 6.62 / 6.57         | 19.70 / 20.06         |
| RMTPP-K             | 35.06     | 16.37     | 0.300     | 6.92 / 6.87         | 17.85 / 18.12         |
| NHP                 | 11.06     | 11.22     | 0.449     | 9.04 / 9.02         | 26.24 / 26.29         |
| ODE                 | 7.54      | 10.14     | 0.492     | 9.48 / 9.46         | 23.54 / 22.96         |
| HYPRO               | 35.69     | 17.21     | 0.295     | 6.63 / 6.61         | 20.58 / 20.53         |
| DeTPP               | 34.32     | 15.84     | 0.260     | **6.03** / **5.98** | 36.88 / 37.18         |
| DeTPP-Hybrid        | **35.77** | **17.27** | **0.237** | **6.03** / **5.98** | **37.08** / **37.20** |

### StackOverflow
| Method              | Acc.      | mAP       | MAE       | OTD  Val / Test       | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:----------|:----------------------|:----------------------|
| MostPopular         | 42.90     | 5.45      | 0.744     | 13.56 / 13.77         | 6.10 / 5.56           |
| RecentHistory       | 26.42     | 5.20      | 0.934     | 14.52 / 14.55         | 8.67 / 6.72           |
| MAE-CE              | 45.41     | 13.00     | 0.641     | 13.57 / 13.64         | 8.78 / 8.31           |
| MAE-CE-K            | 44.85     | 11.16     | 0.644     | 13.41 / 13.51         | 12.42 / 11.42         |
| RMTPP               | 45.43     | 13.33     | 0.701     | 12.95 / 13.17         | 13.26 / 12.72         |
| RMTPP-K             | 44.89     | 11.72     | 0.689     | 12.92 / 13.13         | 14.91 / 14.30         |
| NHP                 | 44.53     | 10.86     | 0.715     | 13.02 / 13.24         | 12.67 / 11.96         |
| ODE                 | 44.38     | 10.12     | 0.711     | 13.04 / 13.27         | 11.37 / 10.52         |
| HYPRO               | 45.18     | 12.88     | 0.715     | 13.04 / 13.26         | 15.57 / 14.69         |
| DeTPP               | 44.81     | 12.36     | 0.665     | **11.96** / **12.05** | **24.06** / **22.54** |
| DeTPP-Hybrid        | **45.64** | **14.08** | **0.638** | 12.06 / 12.19         | 23.47 / 22.22         |

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
