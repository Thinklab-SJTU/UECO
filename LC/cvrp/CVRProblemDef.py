
import torch
import numpy as np
import math

def get_random_problems(batch_size, problem_size):

    depots = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    nodes = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 80
    elif problem_size == 500:
        demand_scaler = 100
    else:
        raise NotImplementedError()
    
    # {50: 40, 100: 50, 200: 80, 500: 100}

    demands = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depots, nodes, demands


def augment_xy_data_by_8_fold(xy_data):
    # xy_data.shape: (batch, N, 2)

    x = xy_data[:, :, [0]]
    y = xy_data[:, :, [1]]
    # x,y shape: (batch, N, 1)

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_xy_data = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    # shape: (8*batch, N, 2)

    return aug_xy_data

def augment_data_rotation(xy_data, num_rotations):
    """
    Perform rotation augmentation on the TSP problem instance to generate
    a total of `num_rotations` samples (including the original data)
    
    params:
        problems: original TSP data (batch, problem, 2)
        num_rotations: number of samples to generate
        
    return:
        aug_problems: The augmented TSP problem data (batch * num_rotations, problem, 2)
    """
    if not isinstance(xy_data, torch.Tensor):
        xy_data = torch.tensor(xy_data, dtype=torch.float32)
    
    batch_size, num_points, _ = xy_data.shape
    
    augmented = []
    
    # x, y coordinates
    x = xy_data[..., 0]  # (batch, problem)
    y = xy_data[..., 1]  # (batch, problem)
    
    # Generate uniformly distributed rotation angles (between 0 and 2\pi), including 0 angles (original data)
    angles = torch.linspace(0, 2 * math.pi, num_rotations)
    
    for angle in angles:
        # Calculate the elements of the rotation matrix
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        
        # perform rotations
        x_rotated = x * cos_theta - y * sin_theta
        y_rotated = x * sin_theta + y * cos_theta
        
        # Merge the rotated coordinates and add them to the augmentation list
        rotated = torch.stack((x_rotated, y_rotated), dim=-1)
        augmented.append(rotated)
    
    # Concatenate all the augmented data (batch * num_rotations, problem, 2)
    aug_xy_data = torch.cat(augmented, dim=0)
    
    return aug_xy_data

def augment_data_hybrid(xy_data, num_rotations: int):
    """
    Hybrid augmentation of TSP problem instances: 
    including 8 kinds of flip transformations 
    and additional rotation transformations
    
    params:
        xy_data: (batch, problem, 2)
        num_rotations: (must >= 8)
        
    return:
        aug_problems: (batch * num_rotations, problem, 2)
    """
    assert num_rotations >= 8, "num_rotations must be greater than 8"
    
    if not isinstance(xy_data, torch.Tensor):
        xy_data = torch.tensor(xy_data, dtype=torch.float32)

    if num_rotations == 8:
        return augment_xy_data_by_8_fold(xy_data)
    
    batch_size, num_points, _ = xy_data.shape
    
    x = xy_data[:, :, [0]]  # (batch, problem, 1)
    y = xy_data[:, :, [1]]  # (batch, problem, 1)
    
    # Generate 8 types of flip transformations (retain the 8 transformations of the original code)
    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)
    
    # Collect all the augmented data and first add 8 kinds of flip transformations
    augmented = [dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8]
    
    num_extra_rotations = num_rotations - 8
    
    # Generate additional rotation angles (from 0 to 2pi, but exclude angles that would repeat with flipping)
    # To avoid repetition with flipping, start generating from a small angle
    start_angle = 0.1
    angles = torch.linspace(start_angle, 2 * math.pi + start_angle, num_extra_rotations + 1)[:-1]
    
    # (batch, problem)
    x_rot = xy_data[..., 0]
    y_rot = xy_data[..., 1]
    
    for angle in angles:
        cos_theta = torch.cos(angle)
        sin_theta = torch.sin(angle)
        
        x_rotated = x_rot * cos_theta - y_rot * sin_theta
        y_rotated = x_rot * sin_theta + y_rot * cos_theta
        
        rotated = torch.stack((x_rotated, y_rotated), dim=-1)
        augmented.append(rotated)
    
    # (batch * num_rotations, problem, 2)
    aug_xy_data = torch.cat(augmented, dim=0)
    
    return aug_xy_data