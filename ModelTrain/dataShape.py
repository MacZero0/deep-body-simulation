# -*- coding: utf-8 -*-
/*
 * @Author: Mac
 * @Email: zhuxin@stu.scu.edu.cn
 * @Date: 2020-12-21 18:06:00
 * @Last Modified by: Mac
 * @Last Modified time: 2020-12-21 18:08:10
 * @Description: Description
 */


import numpy as np
X_DATA = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/X.npy"
Y_DATA = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/Y.npy"
X_CENTER = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/X_centered.npy"
X_MU = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/X_mean_vec.npy"
Y_MU = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/Y_mean_vec.npy"
Y_CENTER = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/Y_centered.npy"
X_COV = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/X_cov.npy"
Y_COV = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/Y_cov.npy"
U_MAT = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/U.npy"
V_MAT = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/V.npy"
Z_MAT = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/Z.npy"
W_MAT = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/W.npy"
ALPHA_VEC = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/alpha.npy"
BETA_VEC = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/beta.npy"
N_MAT = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/norms.npy"
Q_MAT = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/Q.npy"
N_MU = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/norms_mu_vec.npy"
N_CENTER = r"/home/mac/project/projectCopy/DeepCloth/ModelTrain/norms_centered.npy"

X = np.load(X_DATA)
Y = np.load(Y_DATA)
X_center = np.load(X_CENTER)
Y_center = np.load(Y_CENTER)
X_mu = np.load(X_MU)
Y_mu = np.load(Y_MU)
X_cov = np.load(X_COV)
Y_cov = np.load(Y_COV)
U = np.load(U_MAT)
V = np.load(V_MAT)
Z = np.load(Z_MAT)
W = np.load(W_MAT)
alpha = np.load(ALPHA_VEC)
beta = np.load(BETA_VEC)

print("X shape: ",X.shape)
print("Y shape: ",Y.shape)
print("X_cneter shape: ",X_center.shape)
print("Y_center shape: ",Y_center.shape)
print("X_mu shape: ",X_mu.shape)
print("Y_mu shape: ",Y_mu.shape)
print("X_cov shape: ",X_cov.shape)
print("U shape: ",U.shape)
print("V shape: ",V.shape)
print("Z shape: ",Z.shape)
print("W shape: ",W.shape)
print("alpha shape: ",alpha.shape)
print("beta shape: ",beta.shape)


