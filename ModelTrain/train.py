# -*- coding: utf-8 -*-
# /*
#  * @Author: Mac
#  * @Email: zhuxin@stu.scu.edu.cn
#  * @Date: 2020-12-21 18:20:23
#  * @Last Modified by:   Mac
#  * @Last Modified time: 2020-12-21 18:20:23
#  * @Description: Description
#  */


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from NetWork import NetWork
import numpy as np
import json

BATCH_SIZE = 16
WINDOW_SIZE = 32
EPOCHS = 200
LR = 0.0001
LR_DECAY = 0.999
BASE = 128
# LOAD_MODEL = False
LOAD_MODEL = False
Z_MAT = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/Z.npy"
W_MAT = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/W.npy"
ALPHA_VEC = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/alpha.npy"
BETA_VEC = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/beta.npy"
MODEL = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/model.tar"

Z = np.load(Z_MAT)
W = np.load(W_MAT)
alpha = np.load(ALPHA_VEC)
beta = np.load(BETA_VEC) 

Z_tensor = torch.from_numpy(Z).float().cuda()
W_tensor = torch.from_numpy(W).float().cuda()
alpha_tensor = torch.from_numpy(alpha).float().cuda()
beta_tensor = torch.from_numpy(beta).float().cuda()
dt = 1
mae = nn.L1Loss()
mae = mae.cuda()
model = NetWork()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = LR, amsgrad = True)

if LOAD_MODEL:
    checkpoint = torch.load(MODEL)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("model loaded")
    

def compute_batch_loss(indices):

    new_indices = []
    new_batch_size = 0 
    for i in indices.tolist():
        if not (i - 2 < 0 or i + WINDOW_SIZE >= Z.shape[0]):
            new_indices.append(i)
            new_batch_size += 1
    new_indices = torch.tensor(new_indices)
    
    
    r0 = np.random.normal(0, 0.01, size=(new_batch_size, BASE))
    r0 = torch.from_numpy(r0).float().cuda()
    r1 = np.random.normal(0, 0.01, size=(new_batch_size, BASE))
    r1 = torch.from_numpy(r1).float().cuda()
    
    Zi_minus_2_pred = Z_tensor[new_indices] + r0
    Zi_minus_1_pred = Z_tensor[new_indices + 1] + r1
    
    
    alpha_mat = alpha_tensor
    beta_mat = beta_tensor
    
    Z_window_true = torch.Tensor().cuda()
    Z_window_minus_1_true = torch.Tensor().cuda()
    Z_window_pred = torch.Tensor().cuda()
    Z_window_minus_1_pred = torch.Tensor().cuda()
    
    
    
    batch_loss = 0
    
    for i in range(new_batch_size - 1):
        alpha_mat = torch.cat((alpha_mat, alpha_tensor), 0)
        beta_mat = torch.cat((beta_mat, beta_tensor), 0)
            
    
    for i in range(2, WINDOW_SIZE):
        # with torch.no_grad():
        Zi_init = alpha_mat * Zi_minus_1_pred + beta_mat * (Zi_minus_1_pred - Zi_minus_2_pred)
        
        Wt = W_tensor[new_indices + i]
        input_batch = torch.cat((Zi_init, Zi_minus_1_pred, Wt), 1)

        # print(input_batch.shape)
        # print(input_batch)
        Zi_fix = model(input_batch)
        
        Zi_pred = Zi_init + Zi_fix
        

        
        Z_window_true = torch.cat((Z_window_true, Z_tensor[new_indices + i]), 1)
        Z_window_minus_1_true = torch.cat((Z_window_minus_1_true, Z_tensor[new_indices + i - 1]), 1)
        Z_window_pred = torch.cat((Z_window_pred, Zi_pred), 1)
        Z_window_minus_1_pred = torch.cat((Z_window_minus_1_pred, Zi_minus_1_pred), 1)
        
        Zi_minus_2_pred = Zi_minus_1_pred
        Zi_minus_1_pred = Zi_pred
        
    # print(Zi_pred.shape)
    # print(Zi.shape)
    pos_loss = mae(Z_window_pred, Z_window_true)
    # print(pos_loss)
    
    Zi_speed_pred = (Z_window_pred - Z_window_minus_1_pred) / dt
    Zi_speed_true = (Z_window_true - Z_window_minus_1_true) / dt
    
    vel_loss = mae(Zi_speed_pred, Zi_speed_true)
    
    batch_loss = pos_loss + vel_loss
        
    return batch_loss
        
        
    

def train():
        
    min_loss = 10000
    for epoch in range(EPOCHS): 
        epoch_loss = 0
        permutation = torch.randperm(Z.shape[0])   
        batch_loop_cnt = 0
        for batch in range(0, Z.shape[0], BATCH_SIZE):
            optimizer.zero_grad()
            indices = permutation[batch : batch + BATCH_SIZE]
            batch_loss = compute_batch_loss(indices)
            epoch_loss += batch_loss
            batch_loss.backward()
            optimizer.step()
            batch_loop_cnt += 1
        avg_loss = epoch_loss.item() / batch_loop_cnt
        print("epoch ", epoch + 1, ": ", avg_loss)
        if (avg_loss < min_loss):
            min_loss = avg_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": batch_loss,}, MODEL)
            print("loss is the smallest, model saved")
            
            
def write_model_to_json():
    checkpoint = torch.load(MODEL)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("model loaded")
    test_input = torch.from_numpy(np.array([i for i in range(258)]).astype(np.float32)).cuda()
    print(model(test_input))
    to_json = {}
    
    layer_name = ['fc1_weight', 'fc1_bias', 'fc2_weight', 'fc2_bias', 'fc3_weight', 'fc3_bias', 
                  'fc4_weight', 'fc4_bias', 'fc5_weight', 'fc5_bias', 'fc6_weight', 'fc6_bias',
                  'fc7_weight', 'fc7_bias', 'fc8_weight', 'fc8_bias', 'fc9_weight', 'fc9_bias', 
                  'output_weight', 'output_bias']
    
    # test_input = np.array([i for i in range(258)]).astype(np.float32)
    # temp = test_input
    for i, layer in enumerate(model.parameters()):
        # to_json.update({layer_name[i]:layer.data.cpu().numpy().T.tolist()})
        # print(layer_name[])
        # print(layer.data.cpu().numpy().T.shape)
        # print(type(layer.data.cpu().numpy()[0][0]))
        layer_data = layer.data.cpu().numpy().T
        to_json.update({layer_name[i]:layer.data.cpu().numpy().T.tolist()})
    #     print("l: ", layer_data.shape)
    #     if i % 2 == 0:
    #         temp = np.dot(temp, layer_data)
    #     elif i % 2 == 1:
    #         temp = temp + layer_data
    #         if i != 19:
    #             temp = np.maximum(0, temp)
                
        
    #     print("result: ", temp.shape)
    print(to_json['fc9_bias'][3])
                
    # print(temp)
    with open('model_json.json', 'w') as file:
        json.dump(to_json, file)
            
        
if __name__ == "__main__":
    # train()
    write_model_to_json()
            
            
        
    
