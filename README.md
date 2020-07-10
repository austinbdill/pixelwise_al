# Pixelwise Active Learning

This repository is home to a set of Jupyter notebooks and PyTorch code for investigating the use of active learning for monocular depth estimation tasks. 

## Installation

In order to install all relevant packages you must install conda and then run the following commands.

```console
user@device:~$ conda env create -f environment.yml
user@device:~$ conda activate pixelwise
```

## Structure

The code can be best understood by considering each subdirectory and the `main.py` file separately. With the exception of the `notebook` subdirectory, all of the code in this repository is related to training an end-to-end model for monocular depth estimation with active learning. 

#### `notebooks`

The Jupyter notebooks in this folder contain all non-end-to-end approaches to this task including gaussian processes and active learning with a pretrained neural network.  

#### `data`

This folder contains all of the code related to retrieving, cleaning, and building datasets for ingestion by the rest of the code. 

#### `trainers`

This folder contains all code related to constructing, updating, and saving all neural networks used in training. Each file defines a class of trainer that inherits from the BaseTrainer class defined in `base_trainer.py`. 

#### `models`

This folder contains all PyTorch code for defining each neural network and how to sample from neural networks that define a probability distribution. 

#### `configs`

This folder is made up of YAML files that set hyperparameters and the type of training used by the `main.py` driving code. The convention I have used is to name each provided configuration file with an abbreviated version of the corresponding trainer class provided in the `trainers` directory. 

#### `main.py`

This file contains the driver code for the construction of the desired data set and trainer and the training and testing loops for the neural networks.

## Basic Use

## Configuration

## Acknowledgements 
