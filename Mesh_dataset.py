from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *
from scipy.spatial import distance_matrix

import json
import trimesh

class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=15, patch_size=7000):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # i_mesh = self.data_list.iloc[idx][0] #vtk file name
        file_id = self.data_list.iloc[idx, 0]
        mesh_file = f"{file_id}.obj"
        label_file = f"{file_id}.json"

        # Load the mesh using trimesh
        mesh = trimesh.load(mesh_file, process=False)
        points = mesh.vertices
        ids = np.array(mesh.faces)
        cells = points[ids].reshape(-1, 9).astype(dtype='float32')
        # Load labels from the corresponding JSON file
        with open(label_file, 'r') as f:
            labels_data = json.load(f)
        labels_ = np.array(labels_data['labels']).astype('int32').reshape(-1, 1)
        labels = labels_[ids].reshape(-1, 3)
        
        # # new way
        # # move mesh to origin
        # points = mesh.points()
        # mean_cell_centers = mesh.center_of_mass()
        # points[:, 0:3] -= mean_cell_centers[0:3]

        # Normalize mesh data by moving it to the origin and scaling
        # mean_point = points.mean(axis=0)
        
        # std_point = points.std(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        points -= means  # translating all points so the centroid is at the origin

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
            cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
            cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3

        X = cells
        Y = labels

        # initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
        S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')

        # calculate number of valid cells (tooth instead of gingiva)
        positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
        negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

        num_positive = len(positive_idx) # number of selected tooth cells

        if num_positive > self.patch_size: # all positive_idx in this patch
            positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
            selected_idx = positive_selected_idx
        else:   # patch contains all positive_idx and some negative_idx
            num_negative = self.patch_size - num_positive # number of selected gingiva cells
            positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
            negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
            selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

        selected_idx = np.sort(selected_idx, axis=None)
        # selected_idx = np.random.choice(len(X), size=self.patch_size, replace=False)

        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]

        if  torch.cuda.is_available():
            TX = torch.as_tensor(X_train[:, :3], device='cuda')
            TD = torch.cdist(TX, TX)
            D = TD.cpu().numpy()
        else:
            D = distance_matrix(X_train[:, :3], X_train[:, :3])

        S1[D<0.1] = 1.0
        S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, self.patch_size)))

        S2[D<0.2] = 1.0
        S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, self.patch_size)))

        X_train = X_train.transpose(1, 0)
        Y_train = Y_train.transpose(1, 0)

        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                  'A_S': torch.from_numpy(S1), 'A_L': torch.from_numpy(S2)}

        return sample

if __name__ == '__main__':
    dataset = Mesh_Dataset('./train_list_1.csv')
    print(dataset.__getitem__(0))
