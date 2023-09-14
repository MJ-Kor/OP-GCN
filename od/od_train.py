import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import sys
import numpy as np
from barbar import Bar
from torch.utils.tensorboard import SummaryWriter
from ntu_skel_preprocess import get_ntu_skeleton_feature_data
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
    parser.add_argument('--lr', type=float, default=1e-5,
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

net.oc.apply(weights_init_normal)

optimizer = optim.Adam(net.oc.parameters(), lr=args.lr, weight_decay=args.weight_decay)

scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

criterion = nn.CrossEntropyLoss().to(device)

net.train()

try:
    for epoch in range(args.num_epochs):
        total_loss = 0
        train_correct = 0
        for input, target in Bar(train_loader):
            input = input.float().to(device)
            target = target.to(device)
            output = net(input)
            net_loss = criterion(output, target)
            net_loss.backward()
            optimizer.step()
            total_loss += net_loss.item()
            _, predicted = torch.max(output, dim=1)
            train_correct += (target == predicted).sum()
        scheduler.step()
        print('Training Occlusion Detector... Epoch: {}, Loss: {:.3f}, Accuracy: {:.4f}'.format(epoch, total_loss/len(train_loader), train_correct/(len(train_loader)*args.batch_size)))
        if(epoch + 1) % 5 == 0:
            print("*** Validation ***")
            net.eval()
            val_correct = 0
            with torch.no_grad():
                val_loss = 0
                for v_input, v_target in Bar(test_loader):
                    v_input = v_input.float().to(device)
                    v_target = v_target.to(device)
                    v_output = net(v_input)
                    net_val_loss = criterion(v_output, v_target)
                    val_loss += net_val_loss.item()
                    _, val_predicted = torch.max(v_output, 1)
                    val_correct += (v_target == val_predicted).sum()
                print('Validating Occlusion Detector... Epoch: {}, Loss: {:.3f}, Accuracy: {:.4f}'.format(epoch, val_loss/len(test_loader), val_correct/(len(test_loader)*args.batch_size)))
    torch.save(net.sae.state_dict(), "PATH")
    torch.save(net.oc.state_dict(), "PATH")

except KeyboardInterrupt:
    print("Ctrl + C detected. Saving model weights...")
    torch.save(net.sae.state_dict(), "PATH")
    torch.save(net.oc.state_dict(), "PATH")