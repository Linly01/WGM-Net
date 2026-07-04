# WGM-Net: A Multi-Wavelet and Geography-Aware Dynamic Memory Network with Temporal Alignment Mamba for Traffic Flow Forecasting

This is a PyTorch implementation of **WGM-Net**, a multi-wavelet and geography-aware dynamic memory network for traffic flow forecasting.

WGM-Net is designed to capture complex spatio-temporal traffic patterns by integrating multi-wavelet decomposition, holiday-aware temporal embedding, geography-aware dynamic memory, and temporal alignment Mamba.

## Table of Contents

* `config_file`: training and model configuration files for each dataset.
* `dataset`: datasets and processed data files used by the model.
* `lib`: self-defined utility modules, including data loading, preprocessing, normalization, training initialization, and evaluation metrics.
* `model`: implementation of WGM-Net and related model components.
* `preprocessing`: preprocessing scripts, including POI feature construction.
* `run.py`: main script for training and testing.

# Data Preparation

The PEMSD4 and PEMSD8 datasets can be downloaded from the public traffic forecasting benchmark repositories, such as [STSGCN (AAAI-20)](https://github.com/Davidham3/STSGCN).

After downloading the datasets, place the data files in the corresponding directory under:/data

# POI Data Preparation

WGM-Net uses POI features as geographic priors for the GeoDMN module. Since the standard PEMSD4 and PEMSD8 datasets do not directly include POI information, we collect POIs based on the sensor locations.

The sensor IDs and their corresponding longitude and latitude are obtained from the [Caltrans Performance Measurement System](https://pems.dot.ca.gov/).

Based on these sensor coordinates, POI features are collected from OpenStreetMap through the Overpass API using OSMnx.

For each sensor node, POIs within a 500-meter radius are counted and grouped into six functional categories: education, medical, residential, transport, commercial, and leisure.

The detailed category definition is as follows:

| Functional category | Included POI types |
|---|---|
| `education` | school, university, kindergarten |
| `medical` | hospital, clinic, pharmacy |
| `residential` | residential buildings, apartments |
| `transport` | bus station, tram stop, parking, fuel station, taxi facility, traffic signal |
| `commercial` | restaurant, cafe, supermarket, convenience store, bank, ATM, hotel |
| `leisure` | park, sports centre, theatre |

The related scripts are located in:/preprocessing

These POI matrices are used as the geographic prior inputs of the GeoDMN module.

# Requirements

Python 3.6.5, Pytorch 1.9.0, Numpy 1.16.3, argparse, configparser and osmnx

# Model Training
```bash
python run.py --datasets {DATASET_NAME} --mode {MODE_NAME}
```
Replace `{DATASET_NAME}` with one of `PEMSD4`, `PEMSD8`

such as `python run.py --datasets PEMSD4`

There are two options for `{MODE_NAME}` : `train` and `test`

Selecting `train` will retrain the model and save the trained model parameters and records in the `experiment` folder.

With `test` selected, run.py will import the trained model parameters from `{DATASET_NAME}.pth` in the 'pre-trained' folder.
