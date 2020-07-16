# Pixelwise Active Learning

This repository is home to a set of Jupyter notebooks and PyTorch code for investigating the use of active learning for monocular depth estimation tasks. It includes a variety of training methods such as the following (references to relevant slide presentations are listed for each):

- `baseline`: [0](slides/presentation_0.pdf)
- `vanilla`: [1](slides/presentation_1.pdf), [2](slides/presentation_2.pdf)
- `gumbel`: [2](slides/presentation_2.pdf), [3](slides/presentation_3.pdf), [4](slides/presentation_4.pdf), [5](slides/presentation_5.pdf)
- `reinforce`: [3](slides/presentation_3.pdf), [4](slides/presentation_4.pdf), [5](slides/presentation_5.pdf)
- `target`: [3](slides/presentation_3.pdf), [4](slides/presentation_4.pdf)
- `a2c`: [5](slides/presentation_5.pdf)
- `ddpg`: [5](slides/presentation_5.pdf), [6](slides/presentation_6.pdf)

## Installation

In order to install all relevant packages you must install conda and then run the following commands.

```console
user@device:~$ conda env create -f environment.yml
user@device:~$ conda activate pixelwise
```

The data can be found at http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction on the link for the 14GB full data set. It will require an email address for them to send the download link to. Once downloaded, place the resulting two folders in a folder named `kitti` within the data folder. 

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

The basic use for running the code is as follows:

```console
user@device:~$ python main.py --config=<configuration name>.yml
```
In the above `<configuration name>` is to be replaced with the name our your desired configuration file from the `configs` subdirectory. 

## Configuration

The following are some of the configuration options provided by the yaml files in the `configs` subdirectory.

- type: The algorithm / architecture used for training. Expects a string.  
- crop_size: The size of the image subregions to be cropped from the dataset for subsampling. Expects an integer. 
- batch_size: The size of the batches to be used for training and testing. Expects an integer. 
- n_epochs: The number of epochs used for training. Expects an integer. 
- train_period: The number of training steps to be performed between logging. Expects an integer. 
- test_period: The number of testing steps to be performed between logging. Expects an integer. 
- tau: The amount of noise introduced to the Gumbel-Softmax layer. Expects a float. 
- save_output: Whether or not to save image outputs for debugging / presenting results. Expects a boolean. 

## Acknowledgements 

This work was completed while a research assistant and research associate at CMU under the guidance of Barnabas Poczos. In addition, we were advised and funded by an external research group at Lockheed Martin. I am very thankful for the support and interest from both my advisor and those I interacted with at Lockheed Martin. 
