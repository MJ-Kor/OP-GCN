import torch
from torch import optim
import torch.nn.functional as F
import torch.nn as nn
import sys
import numpy as np
from barbar import Bar
from torch.utils.tensorboard import SummaryWriter
from model import OcclusionDetector
from util import weights_init_normal

class TrainerOcclusionDetector:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device
    
    def train(self):
        """ Pretraining the weights for the deep SVDD network using autoencoder"""

        net = OcclusionDetector().to(self.device)

        optimizer = optim.Adam(net.oc.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                    milestones=self.args.lr_milestones, gamma=0.1)
        criterion = nn.CrossEntropyLoss()
        net.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x in Bar(self.train_loader):
                x = x.float().to(self.device)
                
                optimizer.zero_grad()
                x_hat = net(x)
                reconst_loss = criterion(x, x_hat)
                reconst_loss.backward()
                optimizer.step()
                
                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss/len(self.train_loader)))

            if (epoch + 1) % 5 == 0:
                net.eval()
                with torch.no_grad():
                    val_loss = 0
                    for v in Bar(self.test_loader):
                        v = v.float().to(self.device)
                        v_hat = net(v)
                        reconst_loss = criterion2(v, v_hat)
                        val_loss += reconst_loss.item()
                    print('* Validation *')
                    print('Validating Skel_Autoencoder... Epoch: {}, Loss: {:.3f}'.format(epoch, val_loss/len(self.test_loader)))

        torch.save(net.state_dict(), 'PATH')
