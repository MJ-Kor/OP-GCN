import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def cal_distance(joint1, joint2):
    # print("j1:",joint1.shape)
    # print("j2:",joint2.shape)
    return torch.sqrt(torch.sum((joint1-joint2)**2, dim=1))

def bone_difference(left_part, right_part, pose_coordinate):
    symmetric_feature = []
    bone_num = left_part.shape[0]
    batch_size = pose_coordinate.shape[0]
    for i in range(bone_num):
        left_distance = cal_distance(pose_coordinate[:, left_part[i][0]], pose_coordinate[:, left_part[i][1]])
        right_distance = cal_distance(pose_coordinate[:, right_part[i][0]], pose_coordinate[:, right_part[i][1]])
        symmetric_feature.append(torch.abs(left_distance-right_distance).view(batch_size, 1))
    symmetric_feature_torch = torch.cat(symmetric_feature, dim=1)
    return symmetric_feature_torch

def bone_symmetric_feature_extractor(pose_coordinate):
    left_arm = np.array([[5, 6], [6, 7]])-1
    right_arm = np.array([[9, 10], [10, 11]])-1
    left_body = np.array([[21, 9]])-1
    right_body = np.array([[21, 5]])-1
    left_leg = np.array([[1, 13,], [13, 14], [14, 15], [15, 16]])-1
    right_leg = np.array([[1, 17], [17, 18], [18, 19], [19, 20]])-1
    pose_coordinate = pose_coordinate.squeeze(1)
    arm_feature = bone_difference(left_arm, right_arm, pose_coordinate)
    body_feature = bone_difference(left_body, right_body, pose_coordinate)
    leg_feature = bone_difference(left_leg, right_leg, pose_coordinate)
    return torch.cat([arm_feature, body_feature, leg_feature], dim=1)