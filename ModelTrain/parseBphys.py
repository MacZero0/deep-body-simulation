# -*- coding: utf-8 -*-
# @Time    :   2020/12/21 17:51:50
# @FileName:   parseBphys.py
# @Author  :   MacZero0
# @Software:   VSCode

import struct
import os
import numpy as np

cache_dir = r'/home/mac/dataDisk/data/blenderData/blendcache_flag/'
# print(os.path.getsize('506C616E652E303031_000001_00.bphys'))


def get_data():
    X = []
    for file in os.listdir(cache_dir):
        print(file)
        file = os.path.join(cache_dir, file)
        print(os.path.getsize(file) == 34076)
        with open(file ,'rb') as f:
            flatten_points = []
            print([struct.unpack('cccccccc', f.read(8))])
            print(struct.unpack('I', f.read(4)))
            print(struct.unpack('I', f.read(4)))
            print(struct.unpack('I', f.read(4)))
            
            cnt = 20
            type_cnt = 1
            
            while cnt < 34076:
                # print(cnt)
                if type_cnt % 3 != 2 and type_cnt % 3 != 0:
                    x_pos = struct.unpack('f', f.read(4))[0]
                    y_pos = struct.unpack('f', f.read(4))[0]
                    z_pos = struct.unpack('f', f.read(4))[0]
                    #switch z and y to fit Unity
                    flatten_points.append(x_pos)
                    flatten_points.append(z_pos)
                    flatten_points.append(y_pos)
                else:
                    struct.unpack('f', f.read(4))
                    struct.unpack('f', f.read(4))
                    struct.unpack('f', f.read(4))
                cnt += 12
                type_cnt += 1
                
        X.append(flatten_points)
        
    X = np.array(X)
    print(X.shape)
    np.save("X.npy", X)
    
def print_data():
    for file in os.listdir(cache_dir):
        print(file)
        file = os.path.join(cache_dir, file)
        print(os.path.getsize(file) == 34076)
        with open(file ,'rb') as f:
            flatten_points = []
            print([struct.unpack('cccccccc', f.read(8))])
            print(struct.unpack('I', f.read(4)))
            print(struct.unpack('I', f.read(4)))
            print(struct.unpack('I', f.read(4)))
            
            cnt = 20
            type_cnt = 1
            vertex_cnt = 0
            
            while cnt < 34076:
                # print(cnt)
                if type_cnt % 3 != 2 and type_cnt % 3 != 0:
                    x_pos = struct.unpack('f', f.read(4))[0]
                    y_pos = struct.unpack('f', f.read(4))[0]
                    z_pos = struct.unpack('f', f.read(4))[0]
                    #switch z and y to fit Unity
                    flatten_points.append(x_pos)
                    flatten_points.append(z_pos)
                    flatten_points.append(y_pos)
                    if vertex_cnt < 10:
                        print(vertex_cnt , " x: ", x_pos)
                        print(vertex_cnt , " y: ", z_pos)
                        print(vertex_cnt , " z: ", y_pos)
                    vertex_cnt += 1
                # print(vertex_cnt)
                else:
                    struct.unpack('f', f.read(4))
                    struct.unpack('f', f.read(4))
                    struct.unpack('f', f.read(4))
                cnt += 12
                type_cnt += 1
        break
                

    
if __name__ == "__main__":
    get_data()
    print_data()
        

        
