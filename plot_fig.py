import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
# from Mesh_dataset import *
# from meshsegnet import *
from losses_and_metrics_for_mesh import *
import utils
import pandas as pd
import shutil
import csv
import matplotlib.pyplot as plt

if __name__ == '__main__':

    use_visdom = True # if you don't use visdom, please set to False


    # ,loss,DSC,SEN,PPV,val_loss,val_DSC,val_SEN,val_PPV
    path = "losses_metrics_vs_epoch.csv"
    # csv_reader = csv.reader(open(path))
    # for row in csv_reader:
    #     print(row)
    dating_df = pd.read_csv(path,sep=',')
    # print(dating_df)
    # ,loss,DSC,SEN,PPV,val_loss,val_DSC,val_SEN,val_PPV
    index = dating_df.index.values
    # print(index)
    loss = dating_df["loss"].values
    DSC = dating_df["DSC"].values
    SEN = dating_df["SEN"].values
    PPV = dating_df["PPV"].values
    val_loss =dating_df["val_loss"].values
    val_DSC =dating_df["val_DSC"].values
    val_SEN =dating_df["val_SEN"].values
    val_PPV =dating_df["val_PPV"].values
    # print(len(index))
    # print(len(loss))
    plt.plot(index,loss,index,val_loss)
    plt.xlabel("epoches")
    plt.ylabel("loss")
    plt.title("Loss")
    plt.savefig("loss_epoches.png")
