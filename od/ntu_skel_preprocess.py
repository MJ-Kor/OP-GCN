import torch
import numpy as np
import argparse 
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from PIL import Image
from tqdm import tqdm

from util import global_contrast_normalization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class NTU_SKEL_loader(data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y    

def get_ntu_skeleton_feature_data(args, data_dir='PATH'):
    data_dir = 'PATH'
    my_data = np.load(data_dir)
    print(my_data.files)

    # my_data = torch.Tensor(my_data)

    # my_data = my_data.unsqueeze(1)
    # train_per = int(my_data.shape[0]*0.8)

    # train = my_data[:train_per]
    # test = my_data[train_per:]
    print("left_arm")
    train = my_data['train_left_arm'].reshape(-1, 1, 25, 3)
    test = my_data['test_left_arm'].reshape(-1, 1, 25, 3)
    train_target = my_data['train_target_left_arm']
    test_target = my_data['test_target_left_arm']
    # #=============================================================
    # print("right_arm")
    # train = my_data['train_right_arm'].reshape(-1, 1, 25, 3)
    # test = my_data['test_right_arm'].reshape(-1, 1, 25, 3)
    # train_target = my_data['train_target_right_arm']
    # test_target = my_data['test_target_right_arm']
    # #=============================================================
    # print("left_leg")
    # train = my_data['train_left_leg'].reshape(-1, 1, 25, 3)
    # test = my_data['test_left_leg'].reshape(-1, 1, 25, 3)
    # train_target = my_data['train_target_left_leg']
    # test_target = my_data['test_target_left_leg']
    # #=============================================================
    # print("right_leg")
    # train = my_data['train_right_leg'].reshape(-1, 1, 25, 3)
    # test = my_data['test_right_leg'].reshape(-1, 1, 25, 3)
    # train_target = my_data['train_target_right_leg']
    # test_target = my_data['test_target_right_leg']
    # #=============================================================
    # print("body")
    # train = my_data['train_body'].reshape(-1, 1, 25, 3)
    # test = my_data['test_body'].reshape(-1, 1, 25, 3)
    # train_target = my_data['train_target_body']
    # test_target = my_data['test_target_body']
    # #=============================================================
    # print("non occluded")
    # train = my_data['train_none'].reshape(-1, 1, 25, 3)
    # test = my_data['test_none'].reshape(-1, 1, 25, 3)
    # train_target = my_data['train_target_none']
    # test_target = my_data['test_target_none']

    train = torch.Tensor(train)
    test = torch.Tensor(test)
    train_target = torch.LongTensor(train_target)
    test_target = torch.LongTensor(test_target)

    data_train = NTU_SKEL_loader(train, train_target)
    data_test = NTU_SKEL_loader(test, test_target)

    dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                  shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
                                 shuffle=True)
    

    return dataloader_train, dataloader_test
































































###################################################################

# # /home/vimlab/workspace/source/PyTorch-Deep-SVDD/data/NTU/ntu_skeleton_feature.npy
# # /home/vimlab/workspace/source/PyTorch-Deep-SVDD/code_test/Our_ntu_skeleton_data_with_synthetic.npz
# def get_ntu_skeleton_feature_data(args, data_dir='/home/vimlab/workspace/source/PyTorch-Deep-SVDD_real/data/NTU/Our_ntu_skeleton_data_with_synthetic.npz', only_xyz=True, normalize=False):
#     my_data = np.load(data_dir)
#     # if only_xyz == True:
#     #     my_data = my_data[:,:,0:3]
#     # if normalize == True:
#     #     my_data = normalize_joint_coordinates(my_data)

#     # my_data = torch.Tensor(my_data)

#     # my_data = my_data.unsqueeze(1)
#     # train_per = int(my_data.shape[0]*0.8)
#     # test_per = my_data.shape[0]-train_per

#     # train = my_data[:train_per]
#     # test = my_data[train_per:]

#     # train_label = torch.zeros(train.shape[0])
#     # test_label = torch.zeros(test.shape[0])

#     # train = my_data['train']
#     # test = my_data['synthetic_test']

#     train = normalize_joint_location(my_data['train'])
#     test = normalize_joint_location(my_data['synthetic_test'])
#     train = normalize_joint_coordinates(train)
#     test = normalize_joint_coordinates(test)

#     train_label = torch.zeros(train.shape[0])
#     test_label = torch.zeros(test.shape[0])
#     test_label[:4000]=1

#     # print(train.shape)
#     # print(test.shape)
#     # print(train_label.shape)
#     # print(test_label.shape)
#     # print(list(test_label.numpy()).count(1))

#     data_train = NTU_SKEL_loader(train, train_label)
#     data_test = NTU_SKEL_loader(test, test_label)


#     dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
#                                   shuffle=True)
#     dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
#                                  shuffle=True)
    

#     return dataloader_train, dataloader_test
    
# def normalize_joint_location(data):
#     n, j, _ = data.shape
#     normalized_data = data
#     for i in tqdm(range(n), desc="Joint loacation normalizing"):
#         moving_point = data[i][0]
#         normalized_data[i]=normalized_data[i]-moving_point
#     #print(normalized_data[1])
#     return normalized_data

# def normalize_joint_coordinates(data):
#     n, j, c = data.shape
#     x = data[:,:,0].copy().flatten()
#     y = data[:,:,1].copy().flatten()
#     z = data[:,:,2].copy().flatten()
#     mean_x = data[:,:,0].flatten().mean()
#     mean_y = data[:,:,1].flatten().mean()
#     mean_z = data[:,:,2].flatten().mean()
#     std_x = data[:,:,0].flatten().std()
#     std_y = data[:,:,1].flatten().std()
#     std_z = data[:,:,2].flatten().std()

#     x_hat = (x-mean_x) / std_x
#     y_hat = (y-mean_y) / std_y
#     z_hat = (z-mean_z) / std_z

#     x_hat = x_hat.reshape(-1, 25, 1)
#     y_hat = y_hat.reshape(-1, 25, 1)
#     z_hat = z_hat.reshape(-1, 25, 1)

#     normalized_data = np.concatenate([x_hat, y_hat, z_hat], axis=2)
#     return normalized_data
    




            


# dataloader_train, dataloader_test = get_ntu_skeleton_feature_data2(args=args)

# for x in (dataloader_train):
#     print(x.shape)
#     break

# custom_dataset = NTU_SKEL_loader(data)

# dataloader_train = DataLoader(custom_dataset, batch_size=2)

# for x in dataloader_train:
#     x = x.to(device)
#     print(x.get_device())