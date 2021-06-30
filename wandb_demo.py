#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os, random
import wandb
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, roc_auc_score
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=-1)
parser.add_argument("--accumulate_grad_batches", type=int, default=1)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--max_epochs", type=int, default=20)
parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--eval_batch_size", type=int, default=128)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--weight_decay", type=float, default=1e-3)
parser.add_argument("--save_weights", type=str, default="")
parser.add_argument(
    "--model",
    type=str,
    default="svm",
    choices=["svm", "model2"],
)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument(
    "--wandb_mode",
    type=str,
    default="online",
    choices=["online", "offline", "disabled"],
)
hparams = vars(parser.parse_args())

# set seed
if hparams["seed"] < 0:
    hparams["seed"] = random.randint(0, 9999)
    
if __name__ == "__main__":
    wandb.init(project="park-motor", mode=hparams["wandb_mode"])
    wandb.config.update(hparams)

    print(hparams)


# In[5]:


import wandb
wandb.init(project='park_motor', entity='mislam6')


# In[ ]:




