import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_PROGRAM"] = "multimodal_driver.py"

DEVICE = torch.device("cuda:0")
#DEVICE = torch.device("cpu")

# DATASET SETTING
INPUT_DIM = 128

#Files
DATA_DIR = "E:/Saiful/park_motor/data"
TEST_SPLIT = 0.05
DEV_SPLIT = 0.05
