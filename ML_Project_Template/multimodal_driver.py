from __future__ import absolute_import, division, print_function

import argparse
import csv
import os
import random
import pickle
import sys
import numpy as np
from typing import *
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss, BCELoss, MarginRankingLoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from ranking_model import FeedForwardSiamese
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

from argparse_utils import str2bool, seed
from global_configs import DATA_DIR, DEV_SPLIT, TEST_SPLIT, DEVICE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    choices=["saloni-only"], default="saloni-only")
parser.add_argument("--train_batch_size", type=int, default=64)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=50)
parser.add_argument("--dropout_prob", type=float, default=0.5)
parser.add_argument("--margin", type=float, default=0.3)
#Number of output neurons in hidden layer for the ff-siamese model
parser.add_argument("--ff_hidden_dim", type=int, default=64)
parser.add_argument("--scheduler", type=str, choices=["step-lr", "reduce-on-plateau"], default="step-lr")
parser.add_argument("--scheduler_step_size", type=int, default=1)
parser.add_argument("--scheduler_factor", type=float, default=0.80)
parser.add_argument(
    "--model",
    type=str,
    choices=["ff-siamese"],
    default="ff-siamese",
)
parser.add_argument("--learning_rate", type=float, default=0.00001)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--seed", type=seed, default="random")


args = parser.parse_args()
#args.seed = 1486

def return_unk():
    return 0

def get_file_id_from_number(tensor_values, split_name):
    '''
    we converted file_id to a number using LabelEncoder()
    now, we want to get back to that file_id
    tensor_value: list of numbers that replaced the file_id
    split_name: train, test, dev

    returns a list of strings (file_ids)
    '''
    le = preprocessing.LabelEncoder()
    le.classes_ = np.load('file_id_to_numbers_%s.npy'%(split_name), allow_pickle=True)
    file_ids = le.inverse_transform(tensor_values)
    return file_ids

def get_appropriate_dataset(data, split_name):
    '''
    data: pandas dataframe
    split_name: train, test, dev

    returns a tensor dataset
    '''
    #Drop 0.5 labels
    #data = data[data["label"]!=0.5]

    #Redefine labels
    #x1<x2: label 0->-1; x1=x2: label 0.5->0; x1>x2: label 1->1
    data['label'] = -1.0 + 2.0*data['label']


    #Convert file id strings to encoded numbers and return as tensors
    le = preprocessing.LabelEncoder()
    concat_data = list(set(list(data["id1"]) + list(data["id2"])))
    le.fit(concat_data)
    np.save('file_id_to_numbers_%s.npy'%(split_name), le.classes_)
    all_id1 = torch.as_tensor(le.transform(data["id1"]))
    all_id2 = torch.as_tensor(le.transform(data["id2"]))

    all_features1 = torch.tensor(data["features1"], dtype=torch.float)
    all_features2 = torch.tensor(data["features2"], dtype=torch.float)
    all_label = torch.tensor(data["label"], dtype=torch.float)

    #print(all_label)

    #To get back to ids
    #file_ids = get_file_id_from_number(all_id1, "train")
    
    dataset = TensorDataset(
        all_id1,
        all_id2,
        all_features1,
        all_features2,
        all_label
    )

    #print(dataset.tensors[0])
    #print(dataset.tensors[0].size(0))
    #print(dataset.__len__())
    #print(dataset.__getitem__(0))

    return dataset


def set_up_data_loader():
    #Update filename
    filename = ""
    
    if args.dataset=="saloni-only":
        filename = os.path.join(DATA_DIR,"task2_ranking_all_dataset.pkl")
        
    with open(filename, "rb") as handle:
        data = pickle.load(handle)

    train, dev, test = data["train"], data["dev"], data["test"]
    
    train = train.reset_index()
    test = test.reset_index()
    dev = dev.reset_index()

    train_dataset = get_appropriate_dataset(train, "train")
    dev_dataset = get_appropriate_dataset(dev, "dev")
    test_dataset = get_appropriate_dataset(test, "test")
    
    #print(len(train_dataset), len(test_dataset), len(dev_dataset))
    
    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size /
            args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed: int):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    print("Seed: {}".format(seed))

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return

def prep_for_training(num_train_optimization_steps: int):

    if args.model == "ff-siamese":
        model = FeedForwardSiamese(args)

    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    
    scheduler = None
    if args.scheduler == "step-lr":
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_factor)
    elif args.scheduler == "reduce-on-plateau":
        scheduler = ReduceLROnPlateau(optimizer, factor=args.scheduler_factor)

    return model, optimizer, scheduler

def rankingLoss(y1_preds, y2_preds, labels):
    '''
    e = (y==0) ? max(0, |y1_pred - y2_pred| - args.margin) : max(0, -label(y1_pred - y2_pred)+args.margin)
    returns mean error
    '''
    outputs = y1_preds - y2_preds
    zero_indices = torch.where(labels==0.0)
    non_zero_indices = torch.where(labels!=0.0)

    #outputs_non_zero = outputs[non_zero_indices]
    labels_non_zero = labels[non_zero_indices]
    y1_preds_non_zero = y1_preds[non_zero_indices]
    y2_preds_non_zero = y2_preds[non_zero_indices]

    outputs_zero = outputs[zero_indices]
    labels_zero = labels[zero_indices]

    #NaN issue -- fix it
    loss_function = MarginRankingLoss(margin=args.margin)
    L1 = loss_function(y1_preds_non_zero, y2_preds_non_zero, labels_non_zero)*len(labels_non_zero)
    #print("Training Loss: \n")
    #print(L1)
    L2 = torch.sum(torch.maximum(torch.abs(outputs_zero)-args.margin, labels_zero))
    #print(L2)
    L = (L1+L2)/len(labels)
    #print(L)
    #print("--"*10)
    return L

#Ongoing
def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    #Model working on training mode
    model.train()
    tr_loss = 0
    num_tr_examples, num_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        ids1, ids2, features1, features2, labels = batch
        
        y1_preds, y2_preds = model(
            features1,
            features2
        )
        
        #Compute Loss
        loss = rankingLoss(y1_preds, y2_preds, labels)
        #loss_function = MSELoss()
        #loss = loss_function(outputs.flatten(), labels.flatten())

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        num_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()

            if args.scheduler=='step-lr':
                scheduler.step()
            elif args.scheduler=='reduce-on-plateau':
                scheduler.step(tr_loss)
            optimizer.zero_grad()

    return (tr_loss / num_tr_steps)


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, optimizer):
    model.eval()
    dev_loss = 0
    num_dev_examples, num_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)
            ids1, ids2, features1, features2, labels = batch

            y1_preds, y2_preds = model(
            features1,
            features2
            )
        
            #Compute Loss
            loss = rankingLoss(y1_preds, y2_preds, labels)
            #loss_function = MSELoss()
            #loss = loss_function(outputs.flatten(), labels.flatten())


            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            num_dev_steps += 1

    return dev_loss / num_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)
            ids1, ids2, features1, features2, label_ids = batch

            y1_preds, y2_preds = model(
            features1,
            features2
            )

            y_preds = y1_preds - y2_preds
            outputs = y_preds
            outputs[torch.where(torch.abs(y_preds)<args.margin)] = 0.0
            outputs[torch.where(y_preds>args.margin)] = 1.0
            outputs[torch.where(y_preds<args.margin)] = -1.0

            outputs = outputs.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            outputs = np.squeeze(outputs).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(outputs)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)
    
    #n = len(preds)
    #for i in range(0,n):
    #    print(preds[i], y_test[i])

    #non_zeros = np.array(
    #    [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    #preds = preds[non_zeros]
    #y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    #preds = preds >= 0
    #y_test = y_test >= 0

    #f_score = f1_score(y_test, preds, average="weighted")
    #acc = accuracy_score(y_test, preds)

    #return acc, mae, corr, f_score
    return mae, corr


def show_predictions(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)
    
    n = len(preds)
    for i in range(0,n):
        print(preds[i], y_test[i])

    return preds, y_test


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
):
    valid_losses = []
    test_maes = []

    for epoch_i in range(int(args.n_epochs)):
        #print("Before")
        #for param in model.parameters():
        #    print(param)
        #print("================")
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        #print("After")
        #for param in model.parameters():
        #    print(param)
        #print("=============")
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        test_mae, test_corr = test_score_model(
            model, test_data_loader
        )

        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_mae:{}".format(
                epoch_i, train_loss, valid_loss, test_mae
            )
        )

        valid_losses.append(valid_loss)
        test_maes.append(test_mae)

        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "best_valid_loss": min(valid_losses),
                    "best_test_mae": max(test_maes),
                }
            )
        )

    show_predictions(model, test_data_loader)
    return


def main():
    wandb.init(project="finger_tap_dev_v2")
    wandb.config.update(args)
    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )

    return


if __name__ == "__main__":
    main()
