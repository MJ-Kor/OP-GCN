import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from bone_symmetric_feature_extractor import bone_symmetric_feature_extractor
from joint_angle_feature_extractor import joint_angle_feature_extractor
from model import OcclusionDetector


class OcclusionDetection(nn.Sequential):
    def __init__(self,):
        super(OcclusionDetection, self).__init__()

        self.od = OcclusionDetector()
    
    def forward(self, input):
        # (B, C, S, J, N) -> (N, B, S, J, C)
        B, C, S, J, N = input.shape
        input = input.permute(4, 0, 2, 3, 1)

        # Divide body1, body2  (B, S, J, C)
        body1 = einops.rearrange(input[0].clone(), 'B S J C -> B S (J C)')#.view(B, S, J*C)
        body2 = einops.rearrange(input[1].clone(), 'B S J C -> B S (J C)')#.view(B, S, J*C)

        # body1, 2 Occlusion Detection
        pred = []
        for b in range(body1.shape[0]):
            body1_input = body1[b]
            body2_input = body2[b]
            sum1_dim_1 = torch.sum(body1_input, dim=1)
            index = torch.where(sum1_dim_1 != 0)[0]
            trimmed_body1_input = body1_input[index.tolist(), :].view(-1, J, C).unsqueeze(1)
            print("trimmed input:",trimmed_body1_input.shape)
            output1 = self.od1(trimmed_body1_input)
            output_mean1 = torch.mean(output1, dim=0)
            pred1 = torch.sigmoid(output_mean1)
            pred1_label = torch.where(pred1 >= 0.5)[0]
            if int(torch.sum(body2_input))!=0:
                sum2_dim_1 = torch.sum(body2_input, dim=1)
                index = torch.where(sum2_dim_1 != 0)[0]
                trimmed_body2_input = body2_input[index.tolist(), :].view(-1, J, C).unsqueeze(1)
                output2 = self.od1(trimmed_body2_input)
                output_mean2 = torch.mean(output2, dim=0)
                pred2 = torch.sigmoid(output_mean2)
                pred2_label = torch.where(pred2 >= 0.5)[0]
            else:
                pred2_label = None
            
            if pred2_label==None:
                pred.append(pred1_label)
            else:
                pred.append(torch.unique(torch.cat([pred1_label, pred2_label])))

        return input.permute(1, 4, 2, 3, 0), pred


input = torch.randn(16, 3, 64, 25, 2)
print(input.shape)

od_p = OcclusionDetection()
data, label = od_p(input)
print("data:",data.shape)
print("label:", label)

        

            
            
            


        


