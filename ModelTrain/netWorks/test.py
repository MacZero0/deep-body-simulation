# -*- coding: utf-8 -*-
# /*
#  * @Author: Mac
#  * @Email: zhuxin@stu.scu.edu.cn
#  * @Date: 2020-12-21 18:20:15
#  * @Last Modified by:   Mac
#  * @Last Modified time: 2020-12-21 18:20:15
#  * @Description: Description
#  */


import torch 
import torch.nn as nn
import numpy as np
from NetWork import NetWork
import torch.optim as optim

MODEL = r"/home/mac/project/projectCopy/DeepCloth/Data/modelData/model.tar"
LR = 0.0001

mae = nn.L1Loss()

test_pred = torch.FloatTensor([[3, 4], [5, 6], [1, 2], [3, 2]])
test_true = torch.FloatTensor([[1, 3], [2, 2], [6, 2], [3, 0]])

# print(mae(test_pred, test_true))

test_pred = torch.FloatTensor([[3, 4, 5, 6], [1, 2, 3, 2]])
test_true = torch.FloatTensor([[1, 3, 2, 2], [6, 2, 3, 0]])

print(test_pred[0][0])
# print(mae(test_pred, test_true))

print(10 ** (1/400))

a = [0, 0, 1]
b = [0, 1, 0]
print(np.cross(a, b))
print(np.cross(b, a))



Z_MAT = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/Z.npy"
W_MAT = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/W.npy"
ALPHA_VEC = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/alpha.npy"
BETA_VEC = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/beta.npy"
X_MU = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/X_mean_vec.npy"
Y_MU = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/Y_mean_vec.npy"
U_MAT = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/U.npy"
flagInfoData = r"/home/mac/project/projectCopy/DeepCloth/Data/flagInfoData/model_info.txt"
print(flagInfoData)

model = NetWork()
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr = LR, amsgrad = True)
checkpoint = torch.load(MODEL)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
print("model loaded")

isTriangle = False
inp = []
with open(flagInfoData, 'r') as file:
    for line in file:
        if line == 'separate\n':
            isTriangle = True
        elif isTriangle:
            pass
        elif (not isTriangle):
            inp.append(float(line))
            
inp = np.array(inp).astype(np.float32)
X_mu_vec = np.load(X_MU).astype(np.float32)
W = np.load(W_MAT).astype(np.float32)
w0 = W[0]
print("w0 shape: ",w0.shape)
inp = inp - X_mu_vec
U = np.load(U_MAT).astype(np.float32)
inp = np.dot(inp, U)
print("inp1: ")
print(inp)
print('inp1 shape: ', inp.shape)

alpha = np.squeeze(np.load(ALPHA_VEC).astype(np.float32))
beta = np.squeeze(np.load(BETA_VEC).astype(np.float32))

init = alpha * inp + beta * (inp - inp)

print("init: ")
print(init)
print("init shape: ",init.shape)

# inp = [0.5600986,59.26706,12.87532,-0.4832323,2.241695,-4.762509,-0.01744957,2.159874,2.121736,0.3100255,0.8893739,0.2921354,-1.102159,-0.1227315,0.21624,0.1150353,-0.6540993,0.3686737,0.1232128,-0.1803462,-0.2397839,-0.0295306,0.216661,-0.3492311,-0.2230098,-0.21492,0.151893,0.1077328,0.2782298,0.02787836,-0.03173854,-0.05464484,0.07569326,0.02846032,-0.05205929,0.05076176,-0.09909843,0.04711072,0.002273856,0.08338673,-0.08405197,0.02504905,0.01374148,0.005729254,0.002635674,-0.03568163,-0.04918174,-0.1048664,-0.06677486,-0.06572166,0.05800616,0.04828349,-0.08604646,-0.04536181,0.01014021,0.09189994,-0.02360364,-0.02146179,0.002250287,-0.06471324,0.02104639,0.03699034,0.04428532,-0.03045417,-0.03298874,0.01061871,-0.08257756,-0.02820235,0.003827617,-0.01955611,0.03241133,0.009172836,-0.03207607,-0.004441905,0.07718796,0.02214154,0.003927391,-0.007768151,0.01687111,-0.01222891,-0.01199315,-0.01437849,0.007653222,-0.03117232,-0.006700428,-0.003888926,0.005742469,-0.0008593942,-0.002008962,0.006382544,0.003678765,-0.001661523,-0.01217285,-0.003407993,-0.005256139,0.01282048,-0.01021558,-0.02165264,-0.01716163,-0.001449758,-0.015315,-0.02301461,-0.003450268,-0.01057646,5.906817E-05,0.01092819,-0.01458618,-0.009285132,0.01046054,-0.002266264,0.007048104,-0.0185215,-0.008376106,0.01002796,-0.006159548,-0.003947634,0.0210646,-0.00302889,0.01182956,0.005434444,-0.005849218,0.003847829,0.008765333,-0.004825292,0.009225663,-0.0001774684,-0.01660998,-0.01056875,0.2873663,30.61654,6.697726,-0.2585759,1.201102,-2.601929,-0.009701153,1.211088,1.192385,0.1761437,0.5181103,0.1722686,-0.6363782,-0.07214224,0.1339138,0.07088923,-0.3859807,0.2208622,0.07612393,-0.1135517,-0.1510569,-0.01935098,0.1381606,-0.2261237,-0.1430729,-0.1384675,0.09764615,0.06940974,0.1752085,0.01848214,-0.02140681,-0.03642527,0.05158038,0.01779826,-0.03510615,0.03432058,-0.06618813,0.03220712,0.001525129,0.05777447,-0.05890558,0.018056,0.009679328,0.003994798,0.00186123,-0.02551382,-0.03420244,-0.07279038,-0.04738227,-0.04316962,0.04029903,0.03502265,-0.0640087,-0.03347402,0.007454711,0.06973683,-0.01774845,-0.01581847,0.001762477,-0.04859743,0.01538805,0.02656478,0.03176928,-0.02294394,-0.02463751,0.008205795,-0.06162528,-0.02094854,0.002890928,-0.01433333,0.02526677,0.007134693,-0.02358296,-0.003464027,0.05825832,0.01813101,0.003178205,-0.006210891,0.01229457,-0.01002186,-0.008912258,-0.01102316,0.006176692,-0.02524422,-0.005084199,-0.003115152,0.004434479,-0.0007090334,-0.00158888,0.005226371,0.003056486,-0.001281972,-0.01021834,-0.002707698,-0.004196985,0.01026157,-0.008405806,-0.0166593,-0.01335413,-0.001235763,-0.01247865,-0.01923085,-0.002900134,-0.008709588,5.239518E-05,0.009508736,-0.01168904,-0.007875807,0.008941668,-0.001867745,0.005650985,-0.01454548,-0.006877438,0.008243805,-0.005287188,-0.003171172,0.01804988,-0.002624101,0.009960904,0.004662554,-0.005085753,0.003198551,0.007316959,-0.003783096,0.007802724,-0.0001459383,-0.01439138,-0.008962637,-121.8301,-0.00384956]

z_t0 = torch.from_numpy(np.array(inp).astype(np.float32)).cuda()
z_t1 = torch.from_numpy(np.array(init).astype(np.float32)).cuda()

wt = torch.from_numpy(np.array(w0).astype(np.float32)).cuda()
# print("before test_input shape: ", test_input.shape)
test_input = torch.cat((z_t0,z_t1,wt),0)
# test_input = torch.cat([test_input,w0],0)
print("now test_input shape: ", test_input.shape)
print(model(test_input))
print(model(test_input).shape)