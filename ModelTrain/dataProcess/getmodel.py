# -*- coding: utf-8 -*-
# /*
#  * @Author: Mac
#  * @Email: zhuxin@stu.scu.edu.cn
#  * @Date: 2020-12-21 18:05:47
#  * @Last Modified by:   Mac
#  * @Last Modified time: 2020-12-21 18:05:47
#  * @Description: Description
#  */

data_path = r'/home/mac/dataDisk/data/blenderData/flags.txt'
model_path = '/home/mac/project/projectCopy/DeepCloth/ModelTrain/model.txt'
print(data_path)
data = ''
with open(data_path,'r') as f:
    line = f.readline()
    print(line)
    while line != 'OneObjectSeparate\n':
        print(line)
        data = data + line
        line = f.readline()
f.close()
print(data)
with open(model_path,'w+') as f:
    f.write(data)