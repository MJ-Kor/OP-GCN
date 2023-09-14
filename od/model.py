import torch
import torch.nn as nn
import torch.nn.functional as F
from bone_symmetric_feature_extractor import bone_symmetric_feature_extractor
from joint_angle_feature_extractor import joint_angle_feature_extractor
from util import weights_init_normal
import sys
from torch.utils.tensorboard import SummaryWriter
    
class skel_autoencoder(nn.Module):
    def __init__(self, input_dim=75, hidden_dim1=40, hidden_dim2=25):
        super(skel_autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim1),
        nn.BatchNorm1d(num_features=hidden_dim1),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim1, hidden_dim2),
        nn.BatchNorm1d(num_features=hidden_dim2),
        nn.LeakyReLU()
        )
        
        self.decoder = nn.Sequential(
        nn.Linear(hidden_dim2, hidden_dim1),
        nn.BatchNorm1d(num_features=hidden_dim1),
        nn.LeakyReLU(),
        nn.Linear(hidden_dim1, input_dim),
        )
    
    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.encoder(out)
        out = self.decoder(out)
        out = out.view(x.size(0), 1, 25, 3)
        return out
    
    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

class OcclusionClassifier(nn.Module):
    def __init__(self, input_dim=75, output_dim=6):
        super(OcclusionClassifier, self).__init__()
        
        self.layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim)
        )
    
    def forward(self, input):
        return self.layer(input)

class OcclusionDetector(nn.Sequential):
    def __init__(self,
                 SAE_input_dim=75,
                 SAE_hidden_dim1=40,
                 SAE_hidden_dim2=25,
                 OC_input_dim=96,
                 OC_output_dim=6,
                 SAE_pretrained_weights="PATH",
                 OC_pretrained_weights=None):
                 
        super(OcclusionDetector, self).__init__()

        self.sae = skel_autoencoder(SAE_input_dim, SAE_hidden_dim1, SAE_hidden_dim2)
        self.sae.load_state_dict(torch.load(SAE_pretrained_weights))
        for param in self.sae.parameters():
            param.requires_grad = False
        self.oc = OcclusionClassifier(OC_input_dim, OC_output_dim)
        if OC_pretrained_weights != None:
            self.oc.load_state_dict(torch.load(OC_pretrained_weights), strict=True)
        else:
            self.oc.apply(weights_init_normal)
    def forward(self, input):
        joint_angle_feature = joint_angle_feature_extractor(input)
        bone_symmetric_feature = bone_symmetric_feature_extractor(input)
        reconstructed_input = self.sae(input)
        anomaly_score = reconstructed_input - input
        anomaly_score = anomaly_score.view(-1, 75)
        oc_input = torch.cat([joint_angle_feature, bone_symmetric_feature, anomaly_score], dim=1)

        oc_output = self.oc(oc_input)

        return oc_output
    