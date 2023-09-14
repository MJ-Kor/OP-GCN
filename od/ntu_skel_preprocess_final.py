import torch
import numpy as np
import argparse 
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
from PIL import Image
from tqdm import tqdm
import random
from util import global_contrast_normalization
from utils import feeder_ntu


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size")
args = parser.parse_args()



def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


class NTU_SKEL_loader(data.Dataset):
    def __init__(self, data,target):#, target, transform):
        self.data = data
        self.target = target
        #self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y    



def load_data():
    Feeder = import_class('utils.feeder_ntu.Feeder')
    data_loader = dict()

    
    data_loader['train'] = DataLoader(
        dataset=Feeder(split='train'),
        batch_size= 16,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        worker_init_fn=init_seed)
        
    data_loader['test'] = DataLoader(
        dataset=Feeder(split='test'),
        batch_size= 16,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        worker_init_fn=init_seed)
    
    return data_loader['train'], data_loader['test']


train_data, test_data = load_data()
for x in train_data:
    print(x[0].shape)
    break




# /home/vimlab/workspace/source/PyTorch-Deep-SVDD/data/NTU/ntu_skeleton_feature.npy
# /home/vimlab/workspace/source/PyTorch-Deep-SVDD/code_test/Our_ntu_skeleton_data_with_synthetic.npz
def get_ntu_skeleton_feature_data(args, data_dir='./data/NTU/Our_ntu_skeleton_data_with_synthetic.npz', only_xyz=True, normalize=False):
    my_data = np.load(data_dir)
    # if only_xyz == True:
    #     my_data = my_data[:,:,0:3]
    # if normalize == True:
    #     my_data = normalize_joint_coordinates(my_data)

    # my_data = torch.Tensor(my_data)

    # my_data = my_data.unsqueeze(1)
    # train_per = int(my_data.shape[0]*0.8)
    # test_per = my_data.shape[0]-train_per

    # train = my_data[:train_per]
    # test = my_data[train_per:]

    # train_label = torch.zeros(train.shape[0])
    # test_label = torch.zeros(test.shape[0])

    # train = my_data['train']
    # test = my_data['synthetic_test']

    # train = torch.Tensor(normalize_joint_coordinates(my_data['train'])).unsqueeze(1)
    synthetic = my_data['train'].copy()
    synthetic_test = my_data['test'].copy()
    left_arm = [5, 6, 7, 21, 22]
    right_arm = [9,10,11,23,24]
    leg = [12, 13, 14, 15, 16, 17, 18, 19]
    body = [0,1,2,3,4,8,20]

################## train
    for i in left_arm:
        synthetic[:,i,0] = synthetic[:,i,0]-np.random.uniform(0, 0.1)
        synthetic[:,i,1] = synthetic[:,i,1]-np.random.uniform(0, 0.5)
        synthetic[:,i,2] = synthetic[:,i,2]-np.random.uniform(1, 3)

    # for i in right_arm:
    #     synthetic[:,i,0] = synthetic[:,i,0]-np.random.uniform(0, 0.1)
    #     synthetic[:,i,1] = synthetic[:,i,1]-np.random.uniform(0, 0.5)
    #     synthetic[:,i,2] = synthetic[:,i,2]-np.random.uniform(1, 3)

    # for i in leg:
    #     synthetic[:,i,0] = synthetic[:,i,0]-np.random.uniform(0, 0.1)
    #     synthetic[:,i,1] = synthetic[:,i,1]-np.random.uniform(0, 0.5)
    #     synthetic[:,i,2] = synthetic[:,i,2]-np.random.uniform(1, 3)

    # for i in body:
    #     synthetic[:,i,0] = synthetic[:,i,0]-np.random.uniform(0, 0.1)
    #     synthetic[:,i,1] = synthetic[:,i,1]-np.random.uniform(0, 0.5)
    #     synthetic[:,i,2] = synthetic[:,i,2]-np.random.uniform(1, 3)


########## test
    for i in left_arm:
        synthetic_test[:,i,0] = synthetic_test[:,i,0]-np.random.uniform(0, 0.1)
        synthetic_test[:,i,1] = synthetic_test[:,i,1]-np.random.uniform(0, 0.5)
        synthetic_test[:,i,2] = synthetic_test[:,i,2]-np.random.uniform(1, 3)

    # for i in right_arm:
    #     synthetic_test[:,i,0] = synthetic_test[:,i,0]-np.random.uniform(0, 0.1)
    #     synthetic_test[:,i,1] = synthetic_test[:,i,1]-np.random.uniform(0, 0.5)
    #     synthetic_test[:,i,2] = synthetic_test[:,i,2]-np.random.uniform(1, 3)

    # for i in leg:
    #     synthetic_test[:,i,0] = synthetic_test[:,i,0]-np.random.uniform(0, 0.1)
    #     synthetic_test[:,i,1] = synthetic_test[:,i,1]-np.random.uniform(0, 0.5)
    #     synthetic_test[:,i,2] = synthetic_test[:,i,2]-np.random.uniform(1, 3)

    # for i in body:
    #     synthetic_test[:,i,0] = synthetic_test[:,i,0]-np.random.uniform(0, 0.1)
    #     synthetic_test[:,i,1] = synthetic_test[:,i,1]-np.random.uniform(0, 0.5)
    #     synthetic_test[:,i,2] = synthetic_test[:,i,2]-np.random.uniform(1, 3)
    train = torch.Tensor(my_data['train']).unsqueeze(1)
    train_lr = torch.Tensor(synthetic).unsqueeze(1)
    # test = torch.Tensor(normalize_joint_coordinates(my_data['synthetic_test'])).unsqueeze(1)
    test = torch.Tensor(my_data['test']).unsqueeze(1)
    test_lr = torch.Tensor(synthetic_test).unsqueeze(1)



    train_label = torch.zeros(train.shape[0])
    train_label = torch.zeros(train.shape[0])
    test_label = torch.zeros(test.shape[0])
    test_label = torch.zeros(test.shape[0])
    test_label[:4000]=1

    # print(train.shape)
    # print(test.shape)
    # print(train_label.shape)
    # print(test_label.shape)
    # print(list(test_label.numpy()).count(1))

    data_train = NTU_SKEL_loader(train_lr, train)
    data_test = NTU_SKEL_loader(test_lr, test)


    dataloader_train = DataLoader(data_train, batch_size=args.batch_size,
                                  shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size,
                                 shuffle=True)
    

    return dataloader_train, dataloader_test
    
def normalize_joint_coordinates(data):
    n, j, _ = data.shape
    normalized_data = data
    for i in tqdm(range(n), desc="Joint normalizing"):
        moving_point = data[i][0]
        normalized_data[i]=normalized_data[i]-moving_point
    #print(normalized_data[1])
    return normalized_data
            


# dataloader_train, dataloader_test = get_ntu_skeleton_feature_data(args=args, only_xyz=True, normalize=False)

# for x, y in (dataloader_test):
#     print(x.shape)
#     print(y.shape)
#     break

# custom_dataset = NTU_SKEL_loader(data)

# dataloader_train = DataLoader(custom_dataset, batch_size=2)

# for x in dataloader_train:
#     x = x.to(device)
#     print(x.get_device())