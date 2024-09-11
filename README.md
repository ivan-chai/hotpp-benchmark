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
| MostPopular         | 32.78     | 0.85     | 0.752     | 7.37 / 7.38         | 1.08 / 1.01         |
| RecentHistory (OTD) | 19.61     | 0.87     | 0.924     | 7.40 / 7.44         | 2.47 / 2.47         |
| MAE-CE              | 34.08     | 3.47     | 0.693     | 6.88 / 6.90         | 5.84 / 5.90         |
| MAE-CE-K            | 33.69     | 3.25     | 0.698     | 7.18 / 7.19         | 4.42 / 4.43         |
| RMTPP               | 34.15     | 3.47     | 0.749     | 6.86 / 6.88         | 7.04 / 6.71         |
| RMTPP-K             | 33.63     | 3.24     | 0.749     | 7.10 / 7.11         | 5.82 / 5.52         |
| NHP                 | 35.43     | 3.41     | 0.696     | 6.97 / 6.98         | 5.72 / 5.59         |
| ODE                 | 35.60     | 3.35     | 0.695     | 6.96 / 6.97         | 5.50 / 5.57         |
| HYPRO               | 34.26     | 3.46     | 0.759     | 7.04 / 7.06         | 7.72 / 7.10         |
| DeTPP               | 35.16     | 3.79     | 0.693     | **6.62** / **6.64** | 8.84 / 9.03         |
| DeTPP-Hybrid        | **38.26** | **4.65** | **0.641** | 6.67 / 6.69         | **9.22** / **9.25** |

### MIMIC-IV
| Method              | Acc.      | mAP       | MAE      | OTD  Val / Test       | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:---------|:----------------------|:----------------------|
| MostPopular         | 1.79      | 2.64      | 11.23    | 19.89 / 19.88         | 0.73 / 0.75           |
| RecentHistory (OTD) | 0.93      | 2.65      | 4.79     | 19.77 / 19.74         | 2.34 / 2.41           |
| MAE-CE              | 57.39     | **44.98** | 2.85     | **11.62** / **11.64** | 23.49 / 23.62         |
| MAE-CE-K            | 56.70     | 43.04     | 2.89     | 13.41 / 13.37         | 21.32 / 21.11         |
| RMTPP               | 57.12     | 44.09     | 3.67     | 13.72 / 13.75         | 21.61 / 21.53         |
| RMTPP-K             | 56.27     | 42.19     | 3.30     | 14.87 / 14.84         | 19.52 / 19.58         |
| NHP                 | 23.08     | 16.48     | 6.07     | 19.11 / 19.09         | 9.00 / 9.04           |
| ODE                 | 19.64     | 12.82     | 4.13     | 18.99 / 18.99         | 10.02 / 9.98          |
| HYPRO               | 56.91     | 43.04     | 3.78     | 15.13 / 15.08         | 15.94 / 15.92         |
| DeTPP               | 32.38     | 23.01     | **2.62** | 13.33 / 13.32         | **31.30** / **31.13** |
| DeTPP-Hybrid        | **57.42** | 44.96     | 2.85     | 12.66 / 12.66         | 29.27 / 29.09         |

### Retweet
| Method              | Acc.      | mAP       | MAE       | OTD  Val / Test       | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:----------|:----------------------|:----------------------|
| MostPopular         | 58.50     | 39.85     | 18.82     | 174.9 / 173.5         | 26.54 / 25.90         |
| RecentHistory (OTD) | 50.29     | 35.73     | 21.87     | 152.3 / 150.3         | 29.12 / 29.24         |
| MAE-CE              | 59.95     | 46.53     | 18.27     | 173.3 / 172.7         | 34.58 / 32.80         |
| MAE-CE-K            | 59.55     | 45.09     | **18.21** | 168.6 / 167.9         | 37.11 / 34.73         |
| RMTPP               | 60.07     | 46.81     | 18.46     | 167.5 / 166.6         | 48.75 / 46.01         |
| RMTPP-K             | 59.99     | 46.34     | 18.33     | 164.7 / 163.9         | 49.07 / 46.16         |
| NHP                 | **60.09** | **46.85** | 18.43     | 167.1 / 165.9         | 50.17 / 47.50         |
| ODE                 | 59.94     | 46.66     | 18.38     | 166.5 / 165.4         | 49.46 / 46.58         |
| HYPRO               | 59.87     | 46.69     | 18.75     | 170.6 / 169.7         | 53.47 / 50.67         |
| DeTPP               | 58.02     | 43.73     | 19.06     | **134.2** / **131.2** | 58.53 / 55.37         |
| DeTPP-Hybrid        | 59.94     | 46.34     | 18.30     | 138.8 / 135.8         | **58.92** / **55.53** |

### Amazon
| Method              | Acc.      | mAP       | MAE       | OTD  Val / Test     | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:----------|:--------------------|:----------------------|
| MostPopular         | 33.46     | 9.58      | 0.304     | 7.20 / 7.18         | 9.34 / 9.02           |
| RecentHistory (OTD) | 24.23     | 8.15      | 0.321     | 6.43 / 6.41         | 9.73 / 9.21           |
| MAE-CE              | 35.73     | 17.14     | 0.242     | 6.58 / 6.52         | 22.27 / 22.90         |
| MAE-CE-K            | 35.11     | 16.48     | 0.246     | 6.72 / 6.68         | 22.06 / 22.57         |
| RMTPP               | 35.76     | **17.21** | 0.294     | 6.63 / 6.57         | 20.35 / 20.64         |
| RMTPP-K             | 35.06     | 16.37     | 0.300     | 6.92 / 6.87         | 17.85 / 18.12         |
| NHP                 | 11.02     | 11.20     | 0.449     | 9.04 / 9.02         | 26.20 / 26.25         |
| ODE                 | 7.58      | 10.16     | 0.492     | 9.48 / 9.46         | 23.56 / 23.00         |
| HYPRO               | 35.69     | **17.21** | 0.295     | 6.64 / 6.60         | 21.03 / 21.31         |
| DeTPP               | 34.08     | 15.44     | 0.271     | **6.01** / **5.97** | **36.75** / **37.09** |
| DeTPP-Hybrid        | **35.77** | 17.11     | **0.241** | 6.05 / 5.98         | 36.72 / 37.03         |

### StackOverflow
| Method              | Acc.      | mAP       | MAE       | OTD  Val / Test       | T-mAP  Val / Test     |
|:--------------------|:----------|:----------|:----------|:----------------------|:----------------------|
| MostPopular         | 42.90     | 5.45      | 0.744     | 13.56 / 13.77         | 6.33 / 5.86           |
| RecentHistory (OTD) | 26.42     | 5.20      | 0.934     | 14.52 / 14.55         | 8.67 / 6.72           |
| MAE-CE              | 45.41     | 13.00     | 0.641     | 13.57 / 13.64         | 9.13 / 8.72           |
| MAE-CE-K            | 44.85     | 11.16     | 0.644     | 13.41 / 13.51         | 12.42 / 11.42         |
| RMTPP               | 45.43     | 13.33     | 0.700     | 12.95 / 13.16         | 13.61 / 13.17         |
| RMTPP-K             | 44.89     | 11.72     | 0.689     | 12.92 / 13.13         | 14.91 / 14.30         |
| NHP                 | 44.53     | 10.81     | 0.715     | 13.02 / 13.23         | 12.88 / 12.30         |
| ODE                 | 44.36     | 10.01     | 0.712     | 13.05 / 13.27         | 11.52 / 10.89         |
| HYPRO               | 45.18     | 12.88     | 0.714     | 13.02 / 13.28         | 15.74 / 14.92         |
| DeTPP               | 44.78     | 10.54     | 0.681     | 12.14 / 12.27         | 20.69 / 19.87         |
| DeTPP-Hybrid        | **45.45** | **13.36** | **0.639** | **12.07** / **12.17** | **23.05** / **22.04** |

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
