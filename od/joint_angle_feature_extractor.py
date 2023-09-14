import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA

def angle_between_three_points(a, b, c):
    eps = 1e-7
    ba = a - b
    bc = c - b
    batch_dot_product = torch.bmm(ba.unsqueeze(1), bc.unsqueeze(2)).squeeze(1).squeeze(1)
    batch_norm_ab = LA.norm(ba, dim=1)
    batch_norm_bc = LA.norm(bc, dim=1)
    cos_theta = batch_dot_product / (batch_norm_ab * batch_norm_bc)
    cos_theta[torch.isnan(cos_theta)] = 0
    cos_theta = torch.clamp(cos_theta, -1+eps, 1-eps)
    theta_rad = torch.acos(cos_theta)
    return theta_rad

def joint_angle(part, pose_coordinate):
    angle_feature = []
    angle_num = part.shape[0]
    batch_size = pose_coordinate.shape[0]
    #for b in range(batch_size) 
    if angle_num == 1:
        angle = angle_between_three_points(pose_coordinate[:,part[0][0]], pose_coordinate[:, part[0][1]], pose_coordinate[:, part[0][2]])
        #print(angle.shape)
        return angle.view(batch_size, 1)
    else:
        for i in range(angle_num):
            angle = angle_between_three_points(pose_coordinate[:, part[i][0]], pose_coordinate[:, part[i][1]], pose_coordinate[:, part[i][2]])
            angle_feature.append(angle.view(batch_size, 1))
        return torch.cat(angle_feature, dim=1)

def joint_angle_feature_extractor(pose_coordinate):
    head_angle = np.array([[4, 3, 21]])-1
    left_arm_angle = np.array([[6, 5 ,21], [7, 6, 5]])-1
    right_arm_angle = np.array([[10, 9, 21], [11, 10, 9]])-1
    body_angle = np.array([[3, 21, 2], [21, 2, 1], [17, 1, 13]])-1
    left_leg_angle = np.array([[14, 13, 1], [15, 14, 13], [16, 15, 14]])-1
    right_leg_angle = np.array([[18, 17, 1], [19, 18, 17], [20, 19, 18]])-1
    pose_coordinate = pose_coordinate.squeeze(1)
    head_feature = joint_angle(head_angle, pose_coordinate)
    left_arm_feature = joint_angle(left_arm_angle, pose_coordinate)
    right_arm_feature = joint_angle(right_arm_angle, pose_coordinate)
    body_feature = joint_angle(body_angle, pose_coordinate)
    left_leg_feature = joint_angle(left_leg_angle, pose_coordinate)
    right_leg_feature = joint_angle(right_leg_angle, pose_coordinate)

    return torch.cat([head_feature, left_arm_feature, right_arm_feature, body_feature, left_leg_feature, right_leg_feature], dim=1)

