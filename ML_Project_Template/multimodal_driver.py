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

from torch.nn import CrossEntropyLoss, L1Loss, MSELoss
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
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--dropout_prob", type=float, default=0.5)
#Number of output neurons in hidden layer for the ff-siamese model
parser.add_argument("--ff_hidden_dim", type=int, default=64)
parser.add_argument("--scheduler", type=str, choices=["step-lr", "reduce-on-plateau"], default="step-lr")
parser.add_argument("--scheduler_step_size", type=int, default=1)
parser.add_argument("--scheduler_factor", type=float, default=0.95)
parser.add_argument(
    "--model",
    type=str,
    choices=["ff-siamese"],
    default="ff-siamese",
)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--seed", type=seed, default="random")


args = parser.parse_args()


def return_unk():
    return 0


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


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
    #Convert file id strings to encoded numbers and return as tensors
    le = preprocessing.LabelEncoder()
    le.fit(data["id1"])
    np.save('file_id_to_numbers_%s.npy'%(split_name), le.classes_)
    all_id1 = torch.as_tensor(le.transform(data["id1"]))
    all_id2 = torch.as_tensor(le.transform(data["id2"]))

    all_features1 = torch.tensor(data["features1"], dtype=torch.float)
    all_features2 = torch.tensor(data["features2"], dtype=torch.float)
    all_label = torch.tensor(data["label"], dtype=torch.float)

    #To get back to ids
    #file_ids = get_file_id_from_number(all_id1, "train")

    dataset = TensorDataset(
        all_id1,
        all_id2,
        all_features1,
        all_features2,
        all_label
    )
    return dataset


def set_up_data_loader():
    #Update filename
    filename = ""
    if args.dataset=="saloni-only":
        filename = os.path.join(DATA_DIR,"task2_ranking_dataset.pkl")

    with open(filename, "rb") as handle:
        data = pickle.load(handle)

    train, dev, test = np.split(data.sample(frac=1, random_state=42), [int((1.0 - (TEST_SPLIT+DEV_SPLIT))*len(data)), int((1.0 - TEST_SPLIT)*len(data))])
    
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


def train_epoch(model: nn.Module, train_dataloader: DataLoader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        outputs = model(
            input_ids,
            visual,
            acoustic,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
        )
        logits = outputs[0]
        loss_fct = MSELoss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss / nb_tr_steps


def eval_epoch(model: nn.Module, dev_dataloader: DataLoader, optimizer):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]

            loss_fct = MSELoss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_steps += 1

    return dev_loss / nb_dev_steps


def test_epoch(model: nn.Module, test_dataloader: DataLoader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )

            logits = outputs[0]

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.detach().cpu().numpy()

            logits = np.squeeze(logits).tolist()
            label_ids = np.squeeze(label_ids).tolist()

            preds.extend(logits)
            labels.extend(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

    return preds, labels


def test_score_model(model: nn.Module, test_dataloader: DataLoader, use_zero=False):

    preds, y_test = test_epoch(model, test_dataloader)
    non_zeros = np.array(
        [i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    y_test = y_test >= 0

    f_score = f1_score(y_test, preds, average="weighted")
    acc = accuracy_score(y_test, preds)

    return acc, mae, corr, f_score


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
):
    valid_losses = []
    test_accuracies = []

    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        assert False
        test_acc, test_mae, test_corr, test_f_score = test_score_model(
            model, test_data_loader
        )

        print(
            "epoch:{}, train_loss:{}, valid_loss:{}, test_acc:{}".format(
                epoch_i, train_loss, valid_loss, test_acc
            )
        )

        valid_losses.append(valid_loss)
        test_accuracies.append(test_acc)

        wandb.log(
            (
                {
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "test_acc": test_acc,
                    "test_mae": test_mae,
                    "test_corr": test_corr,
                    "test_f_score": test_f_score,
                    "best_valid_loss": min(valid_losses),
                    "best_test_acc": max(test_accuracies),
                }
            )
        )


def main():
    wandb.init(project="MAG")
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


if __name__ == "__main__":
    main()
