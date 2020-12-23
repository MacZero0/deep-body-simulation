# -*- coding: utf-8 -*-
# /*
#  * @Author: Mac
#  * @Email: zhuxin@stu.scu.edu.cn
#  * @Date: 2020-12-21 18:05:47
#  * @Last Modified by:   Mac
#  * @Last Modified time: 2020-12-21 18:05:47
#  * @Description: Description
#  */


import numpy as np
import json

EXTER_DATA = r"/home/mac/project/projectCopy/DeepCloth/Data/flagInfoData/external_info.json"
X_DATA = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/X.npy"
Y_DATA = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/Y.npy"
X_CENTER = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/X_centered.npy"
X_MU = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/X_mean_vec.npy"
Y_MU = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/Y_mean_vec.npy"
Y_CENTER = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/Y_centered.npy"
X_COV = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/X_cov.npy"
Y_COV = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/Y_cov.npy"
U_MAT = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/U.npy"
V_MAT = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/V.npy"
Z_MAT = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/Z.npy"
W_MAT = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/W.npy"
ALPHA_VEC = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/alpha.npy"
BETA_VEC = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/beta.npy"
N_MAT = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/norms.npy"
Q_MAT = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/Q.npy"
N_MU = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/norms_mu_vec.npy"
N_CENTER = r"/home/mac/project/projectCopy/DeepCloth/Data/parseData/norms_centered.npy"

def json_to_npy(data_dir):
    item_cnt = 1
    Y = []
    with open(data_dir, 'r') as file:
        for item in file:
            data_line = json.loads(item)
            data = data_line[str(item_cnt)]
            rot = data['rot']
            strength = data['strength']
            
            Y.append([rot, strength])
            item_cnt += 1
    

    Y = np.array(Y)
    
    print("Y shape: ", Y.shape)
    
    np.save("Y.npy", Y)
    
def get_mean_vec_and_centered_mat_x():
    X = np.load(X_DATA)
    # Y = np.load(Y_DATA)
    
    X_mu_vec = np.mean(X, axis=0)
    # Y_mu_vec = np.mean(Y, axis=0)
    print(X_mu_vec)
    
    X_centered = X - X_mu_vec
    # Y_centered = Y - Y_mu_vec
    
    print("X_mu_vec shape: ", X_mu_vec.shape)
    # print("Y_mu_vec shape: ", Y_mu_vec.shape)
    print("X_centered shape: ", X_centered.shape)
    # print("Y_centered shape: ", Y_centered.shape)
    
    print(X[58][504])
    print(X_mu_vec[504])
    print(X[58][504] - X_mu_vec[504])
    print(X_centered[58][504])
    
    np.save("X_mean_vec.npy", X_mu_vec)
    # np.save("Y_mean_vec.npy", Y_mu_vec)
    np.save("X_centered.npy", X_centered)
    # np.save("Y_centered.npy", Y_centered)
    
    # for col in range(X_centered.shape[1]):
    #     for row in range(X_centered.shape[0]):
    #         if X_centered[row][col] != 0:
    #             print("not zero")
    print(X_centered)
                
def get_mean_vec_and_centered_mat_y():
    Y = np.load(Y_DATA)
    
    Y_mu_vec = np.mean(Y, axis=0)
    print(Y_mu_vec)
    
    Y_centered = Y - Y_mu_vec
    
    print("Y_mu_vec shape: ", Y_mu_vec.shape)
    print("Y_centered shape: ", Y_centered.shape)
    
    print(Y[58][1])
    print(Y_mu_vec[1])
    print(Y[58][1] - Y_mu_vec[1])
    print(Y_centered[58][1])
    
    np.save("Y_mean_vec.npy", Y_mu_vec)
    np.save("Y_centered.npy", Y_centered)
    
    # for col in range(Y_centered.shape[1]):
    #     for row in range(Y_centered.shape[0]):
    #         if Y_centered[row][col] != 0:
    #             print("not zero")
    print(Y)
                
def get_cov_mat():
    X_centered = np.load(X_CENTER)
    Y_centered = np.load(Y_CENTER)
    X_cov = np.cov(X_centered.T)
    Y_cov = np.cov(Y_centered.T)
    print("X_cov shape: ", X_cov.shape)
    print("Y_cov shape: ", Y_cov.shape)
    np.save("X_cov.npy", X_cov)
    np.save("Y_cov.npy", Y_cov)

def perform_pca(k):
    X_cov = np.load(X_COV)
    Y_cov = np.load(Y_COV)
    print('X cov shape: ', X_cov.shape)
    print('Y cov shape: ', Y_cov.shape)
    eigen_values_x, eigen_vectors_x = np.linalg.eig(X_cov)
    print("eigen vectors x shape: ", eigen_vectors_x.shape)
    sort_idx_x = np.argsort(eigen_values_x)
    
    U = eigen_vectors_x[:, sort_idx_x[-1]].reshape(-1,1)

    for i in range(1, k):
        e_vec = eigen_vectors_x[:, sort_idx_x[-i - 1]].reshape(-1,1)
        # print("e vec shape: ", e_vec.shape)
        U = np.hstack((U, e_vec))
    print("U shape: ", U.shape) 
    
    eigen_values_y, eigen_vectors_y = np.linalg.eig(Y_cov)
    sort_idx_y = np.argsort(eigen_values_y)

    V = eigen_vectors_y[:, sort_idx_y[-1]].reshape(-1,1)
    for i in range(1, len(eigen_values_y)):
        e_vec = eigen_vectors_y[:, sort_idx_y[-i-1]].reshape(-1,1)
        V = np.hstack((V, e_vec))
    print("V shape: ", V.shape) 
        
    np.save("U.npy", U)
    np.save("V.npy", V)
    
    X_centered = np.load(X_CENTER)
    Y_centered = np.load(Y_CENTER)
    
    Z = np.dot(X_centered, U)
    W = np.dot(Y_centered, V)
    
    print("Z shape: ", Z.shape) 
    print("W shape: ", W.shape) 
    
    np.save("Z.npy", Z)
    np.save("W.npy", W)
    
def get_init_model(k):
    alpha = np.zeros(k)
    beta = np.zeros(k)
    Z = np.load(Z_MAT)
    len_t = Z.shape[0]
    
    for i in range(k):
        Zt = Z[2:, i].reshape(1,-1)
        Zt_minus_1 =  Z[1:(len_t - 1), i].reshape(1,-1)
        Zt_minus_2 =  Z[0:(len_t - 2), i].reshape(1,-1)
        Zspeed = Zt_minus_1 - Zt_minus_2
        A_vec = np.vstack((Zt_minus_1, Zspeed)) 
        init_model_vec = np.linalg.lstsq(A_vec.T, Zt.T, rcond=None)[0] 
        alpha[i] = init_model_vec[0][0]
        beta[i] = init_model_vec[1][0]
        
    np.save("alpha.npy", alpha.reshape(1,-1))
    np.save("beta.npy", beta.reshape(1,-1))
    print("alpha shape: ", alpha.shape)
    print("beta shape: ", beta.shape)
    
    
def parse_model_info():
    isTriangle = False
    vertex_cnter = 0
    cnter = 0
    tmp_list = []
    to_json = {}
    to_json.update({'vertex_data':{}})
    to_json.update({'triangle_data':{}})
    
    
    with open('model_info.txt', 'r') as file:
        for line in file:
            if line == 'separate\n':
                isTriangle = True
            
            #triangle data
            elif isTriangle:
                if cnter < 2:
                    tmp_list.append(int(line))
                    cnter += 1
                else:
                    tmp_list.append(int(line))
                    ori_list = tmp_list[:] 
                    
                    for i,v in enumerate(ori_list):
                        tmp_list.pop(i)
                        
                        try:
                            to_append = to_json['triangle_data'][v]
                            to_append.append(tmp_list)
                            to_json['triangle_data'].update({v:to_append})
                        except KeyError:
                            to_json['triangle_data'].update({v:[tmp_list]})
                            
                        tmp_list = ori_list[:]
                        
                    cnter = 0
                    tmp_list = []
                    
            #vertex data
            elif not isTriangle:
                if cnter < 2:
                    tmp_list.append(float(line))
                    cnter += 1
                else:
                    tmp_list.append(float(line))
                    to_json['vertex_data'].update({vertex_cnter:tmp_list})
                    vertex_cnter += 1
                    cnter = 0
                    tmp_list = []
                    
    with open('model_info_dict.json', 'w') as file:
        json.dump(to_json, file)
                    
def sort_triangle_data():
    data_all = {}
    sorted_data = {}
    with open('model_info_dict.json', 'r') as file:
        data_all = json.load(file)
        vertex_data = data_all['vertex_data']
        triangle_data = data_all['triangle_data']
        
        
        for key in triangle_data.keys():
            vert_triangle_data = triangle_data[key]
            ori_vert = vertex_data[key]
            sorted_triangle = []
                   
            for tri in vert_triangle_data:
                vert1 = vertex_data[str(tri[0])]
                vert2 = vertex_data[str(tri[1])]
                sorted_vert = []
                            
                #decide upper or lower triangles 
                #upper
                if (vert1[1] > ori_vert[1] or vert2[1] > ori_vert[1]):
                    if vert1[2] == vert2[2]:
                        if vert1[1] < vert2[1]:
                            sorted_vert.append(tri[0])
                            sorted_vert.append(tri[1])
                        else:
                            sorted_vert.append(tri[1])
                            sorted_vert.append(tri[0])
                    elif vert1[2] < vert2[2]:
                        sorted_vert.append(tri[0])
                        sorted_vert.append(tri[1])
                    else:
                        sorted_vert.append(tri[1])
                        sorted_vert.append(tri[0])
                #lower
                else:
                    if vert1[2] == vert2[2]:
                        if vert1[1] < vert2[1]:
                            sorted_vert.append(tri[1])
                            sorted_vert.append(tri[0])
                        else:
                            sorted_vert.append(tri[0])
                            sorted_vert.append(tri[1])
                    elif vert1[2] < vert2[2]:
                        sorted_vert.append(tri[1])
                        sorted_vert.append(tri[0])
                    else:
                        sorted_vert.append(tri[0])
                        sorted_vert.append(tri[1])   
                        
                sorted_triangle.append(sorted_vert)
            
            
            sorted_data.update({key:sorted_triangle})
            
        data_all.update({'sorted_triangle_data':sorted_data})
        with open('model_sorted_info.json', 'w') as file:
            json.dump(data_all, file)
            
def compute_vertices_normals():
    X = np.load(X_DATA) 
    X = X.astype(np.float32)
    # print(type(X[0][0]))
    sorted_triangles = {}
    vert_norms_all = []
    
    with open('model_sorted_info.json', 'r') as file:
        data_all = json.load(file)
        sorted_triangles = data_all['sorted_triangle_data']
    
    for row in range(X.shape[0]):
        vert_norms_per_row = np.array([])
        for ori_vert_id in range(int(X.shape[1] / 3)):
            vert_triangle_list = sorted_triangles[str(ori_vert_id)]
            ori_vert_pos = X[row][ori_vert_id * 3: (ori_vert_id * 3 + 3)]
            face_nors = []
            for triangle in vert_triangle_list:
                #anti-clockwise, note that numpy uses right handed coordinate, unity uses left handed coordinate
                vert0_id = int(triangle[0])
                vert1_id = int(triangle[1])
                vert0_pos = X[row][vert0_id * 3: (vert0_id * 3 + 3)]
                vert1_pos = X[row][vert1_id * 3: (vert1_id * 3 + 3)]
                
                # if row == 0 and ori_vert_id == 945:
                #     print("ori_vert_pos: ", ori_vert_pos)
                #     print("vert0_pos: ", vert0_pos)
                #     print("vert1_pos: ", vert1_pos)
                
                vec0 = vert0_pos - ori_vert_pos
                vec1 = vert1_pos - ori_vert_pos
                
                face_nors.append(np.cross(vec0, vec1))
            
            # if row == 0 and ori_vert_id == 945:
            face_nors = np.array(face_nors)
            vertex_norm = np.mean(face_nors, axis = 0)
            vertex_norm = vertex_norm / np.linalg.norm(vertex_norm)
            # print(vertex_norm)
            vert_norms_per_row = np.hstack((vert_norms_per_row, vertex_norm))
        print(row)
        # print(vert_norms_per_row)
            
        vert_norms_all.append(vert_norms_per_row)

    vert_norms_all = np.asarray(vert_norms_all)
    np.save("norms.npy", vert_norms_all)
    print(vert_norms_all.shape)        
            # print()
            
def get_mean_centered_norms():
    norms = np.load(N_MAT)
    norms_mu_vec = np.mean(norms, axis=0)
    norms_centered = norms - norms_mu_vec
    np.save("norms_mu_vec.npy", norms_mu_vec)
    np.save("norms_centered.npy", norms_centered)
    print("norms_mu_vec shape: ", norms_mu_vec.shape)
    print("norms_centered shape: ", norms_centered.shape)
    
    Z = np.load(Z_MAT)
    Q = np.linalg.lstsq(Z, norms_centered, rcond=None)[0].T
    
    np.save("Q.npy", Q)
    print("Q shape: ", Q.shape)
    
def write_data_to_json():
    
    # X = np.load(X_DATA).astype(np.float32).tolist()
    # Y = np.load(Y_DATA).astype(np.float32).tolist()
    X_mu_vec = np.load(X_MU).astype(np.float32).tolist()
    # X_centered = np.load(X_CENTER).astype(np.float32).tolist()
    Y_mu_vec = np.load(Y_MU).astype(np.float32).tolist()
    # Y_centered = np.load(Y_CENTER).astype(np.float32).tolist()
    # X_cov = np.load(X_COV).astype(np.float32).tolist()
    # Y_cov = np.load(Y_COV).astype(np.float32).tolist()
    U = np.load(U_MAT).astype(np.float32).tolist()
    V = np.load(V_MAT).astype(np.float32).tolist()
    # Z = np.load(Z_MAT).astype(np.float32).tolist()
    # W = np.load(W_MAT).astype(np.float32).tolist()
    alpha = np.squeeze(np.load(ALPHA_VEC).astype(np.float32)).tolist()
    beta = np.squeeze(np.load(BETA_VEC).astype(np.float32)).tolist()
    Q = np.load(Q_MAT).astype(np.float32).tolist()
    # norms = np.load(N_MAT).astype(np.float32).tolist()
    norms_mu_vec = np.load(N_MU).astype(np.float32).tolist()
    # norms_centered = np.load(N_CENTER).astype(np.float32).tolist()
    
    data = {}
    # data.update({"X": X})
    # data.update({"Y": Y})
    data.update({"X_mu_vec": X_mu_vec})
    # data.update({"X_centered": X_centered})
    data.update({"Y_mu_vec": Y_mu_vec})
    # data.update({"Y_centered": Y_centered})
    # data.update({"X_cov": X_cov})
    # data.update({"Y_cov": Y_cov})
    data.update({"U": U})
    data.update({"V": V})
    # data.update({"Z": Z})
    # data.update({"W": W})
    data.update({"alpha": alpha})
    data.update({"beta": beta})
    data.update({"Q": Q})
    # data.update({"norms": norms})
    data.update({"norms_mu_vec": norms_mu_vec})
    # data.update({"norms_centered": norms_centered})
    
    with open("data_json.json", 'w') as file:
        json.dump(data, file)
        
    print("done")
                
def write_mat_transpose():
    data = {}
    with open("data_json.json", 'r') as file:
        data = json.load(file)
    U_T = np.load(U_MAT).astype(np.float32).T
    print("U_T shape: ", U_T.shape)
    U_T = U_T.tolist()
    data.update({'U_T': U_T})
    Q_T = np.load(Q_MAT).astype(np.float32).T
    print("Q_T shape: ", Q_T.shape)   
    Q_T = Q_T.tolist()
    data.update({'Q_T': Q_T})
    
    with open("data_json.json", 'w') as file:
        json.dump(data, file)
     
        
                        
                
                
                    
                
                    
                
                
    
    
    
            
def debug():
    # X = np.load(X_DATA)
    # print(X[1223][0])
    # with open(EXTER_DATA, 'r') as file:
    #     item_cnt = 0
    #     for item in file:
    #         item_cnt += 1
    #         # if item_cnt == 1224:
    #         item = json.loads(item)
    #         print(item[str(item_cnt)]['rot'])
    # with open('model_info_dict.json', 'r') as file:
    #     data = json.load(file)
    #     # print(data['triangle_data']['945'])
    #     print(len(data['triangle_data'].keys()))
    
    # Z = np.load(Z_MAT)
    # print(Z)
    
    # with open("data_json.json", 'r') as file:
    #     data = json.load(file)
    #     print(data['alpha'][127])
    #     print(data["U"][127][11])
        
    with open("PaiForward.json", 'r') as file:
        data = json.load(file)
        print(len(data['Dense_common_1_weights']))
        print(len(data['Dense_common_1_weights'][0]))
        print(len(data['Dense_common_1_bias']))
        
    print("done")
            

            
    
    
if __name__ == "__main__":
    # json_to_npy(EXTER_DATA)
    # get_mean_vec_and_centered_mat_x()
    # get_mean_vec_and_centered_mat_y()
    # get_cov_mat()
    # perform_pca(128)
    get_init_model(128)
    # debug()
    parse_model_info()
    sort_triangle_data()
    compute_vertices_normals()
    get_mean_centered_norms()
    write_data_to_json()
    write_mat_transpose()
    
    
                
            
