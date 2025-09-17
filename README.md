Unlocking-LoRA-s-Capabilities-for-Fine-tuning-Large-Language-Models-with-Only-Forward-Pass
====================================================

Official implementation of the paper submitted to ICASSP2026: Unlocking LoRA's Capabilities for Fine-tuning Large Language Models with Only Forward Pass


## 1) Overview

This repo contains the source code and reproducing guide of ZOBCD-LoRA.
This research endeavor is designed to help researchers better understand the capabilities, limitations and principles
associated with the BP-free, zeroth-order (ZO) optimization combined with LoRA as a solution for reducing memory costs during Large
Language Model (LLM) fine-tuning. 



## 2) Getting Started

### Environment

To create the environment, all you need is:

```bash
conda create -n zobcd python=3.10
conda activate zobcd
pip install -r requirements.txt
```
Our code uses wandb to monitor the training data, so please ensure that you have completed the configuration of wandb in advance.

### Datasets
Regarding the preparation of the dataset, we strongly recommend downloading each dataset to your local machine first using `datasets/download.py`, and then loading it using `load_from_disk`. Of course, you can also choose to load it directly. Some simple modifications to the paths may be required in `task.py`.

### Pretrained Models
In our code, it is required that the model must be downloaded to the local machine and the folder must be named as follows:

opt-1.3b/   
opt-13b/  
Qwen2-7B-Instruct/     
llama-3.2-3B-Instruct/  
llama-3.1-8B-Instruct/  
llama-2-7b-hf/          

### Quick Start


After properly setting up the model and the datasets, you can run `train.sh` for a quick start.

The `--lora_bcd` parameter is used to control whether to use the ZOBCD-LoRA that we proposed. If this parameter is not specified, MeZO-LoRA will be executed. You can easily conduct a comparison between them and see the result in wandb.

`train.sh` is used for training models in the OPT series, while `train_bf.sh` is used for training models in the LLaMA series and the Qwen model. The main difference is that the latter requires the use of bfloat16 for loading and training.

## 3) Visualization
Our code also provides a convenient script for you to visualize the results in wandb using matplotlib. In the visualization folder, you can first run `get_runid.py` and `get_history.py` to obtain project information, and then use `vis.py` for visualization to reproduce the chart information in our paper.

## 4) Ablation Study
We also provide the relevant code for the more in-depth ablation study conducted in the supplementary materials, which pertains to the magnitude-based gradient selecting algorithm. For a quick start, you firstly make sure the  `--lora_bcd` parameter is set to False. Then set `--selective` and `--selective_ratio` simultaneously. 

If you want to conduct experiments related to the magnitude-based gradient selecting block coordinate descent algorithm, set `--selective`, `--selective_ratio` and `--bcd` simultaneously. 
