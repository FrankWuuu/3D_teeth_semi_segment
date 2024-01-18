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
# import pandas as pd
import shutil
import csv

if __name__ == '__main__':

    use_visdom = False # if you don't use visdom, please set to False

    if use_visdom:
        # set plotter
        global plotter
        plotter = utils.VisdomLinePlotter(env_name="model_name")

    # ,loss,DSC,SEN,PPV,val_loss,val_DSC,val_SEN,val_PPV
    path = "losses_metrics_vs_epoch.csv"
    csv_reader = csv.reader(open(path))
    for row in csv_reader:
        print(row)

    # losses, mdsc, msen, mppv = [], [], [], []
    # val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []

    # # training
    # model.train()
    # running_loss = 0.0
    # running_mdsc = 0.0
    # running_msen = 0.0
    # running_mppv = 0.0
    # loss_epoch = 0.0
    # mdsc_epoch = 0.0
    # msen_epoch = 0.0
    # mppv_epoch = 0.0

    # if use_visdom:
    #     plotter.plot('loss', 'train', 'Loss', epoch+(i_batch+1)/len(train_loader), running_loss/num_batches_to_print)
    #     plotter.plot('DSC', 'train', 'DSC', epoch+(i_batch+1)/len(train_loader), running_mdsc/num_batches_to_print)
    #     plotter.plot('SEN', 'train', 'SEN', epoch+(i_batch+1)/len(train_loader), running_msen/num_batches_to_print)
    #     plotter.plot('PPV', 'train', 'PPV', epoch+(i_batch+1)/len(train_loader), running_mppv/num_batches_to_print)


