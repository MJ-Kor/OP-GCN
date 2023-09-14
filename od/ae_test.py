import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import sys


def ae_eval(net, dataloader, device):
    state_dict = torch.load('PATH')
    
    net.load_state_dict(state_dict)
    net.eval()
    input = None
    pred = None
    with torch.no_grad():
        for idx, x in enumerate(dataloader):
            x = x.float().to(device)
            x_hat = net(x)

            # print(x.shape)
            # print(x_hat.shape)

            if idx == 0:
                input = x.unsqueeze(1)
                pred = x_hat.unsqueeze(1)
            input = torch.cat([input, x.unsqueeze(1)], dim=0)
            pred = torch.cat([pred, x_hat.unsqueeze(1)], dim=0)
    input = input.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    np.save('PATH', pred)
    np.save('PATH', input)
