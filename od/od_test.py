import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import sys
import numpy as np
from barbar import Bar
from torch.utils.tensorboard import SummaryWriter
from ntu_skel_preprocess import get_ntu_skeleton_feature_data
# from ntu_skel_preprocess_final import load_data
from model import OcclusionDetector
from util import weights_init_normal
import argparse 
import torchmetrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument("--num_epochs_ae", type=int, default=10,
                        help="number of epochs for the pretraining")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.5e-6,
                        help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--weight_decay_ae', type=float, default=0.5e-3,
                        help='Weight decay hyperparameter for the L2 regularization')
    parser.add_argument('--lr_ae', type=float, default=1e-4,
                        help='learning rate for autoencoder')
    parser.add_argument('--lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=1024, 
                        help="Batch size")
    args = parser.parse_args() 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_ntu_skeleton_feature_data(args)

    net = OcclusionDetector().to(device=device)

    #net.oc.load_state_dict(torch.load("/home/vimlab/workspace/source/ICRA/weights/OC/OC_weights.pth"))

    net.eval()

    with torch.no_grad():
        test_correct = 0
        for input, target in Bar(test_loader):
            input = input.float().to(device)
            target = target.to(device)
            output = net(input)
            _, predicted = torch.max(output, dim=1)
            test_correct += (target == predicted).sum()
        print('Test Accuracy: {:.4f}'.format(test_correct/(len(test_loader)*args.batch_size)))





