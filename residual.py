import torch
import torch.nn as nn
from CDAE import CDAE
from utils import patch
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import sys 
from patchify import patchify


root_path = 'C:/Users/cy/project/cv/data'
image_dir = os.path.join(root_path,'train_image')
pyramid_path = os.path.join(root_path,"pyramid")


def residual(x,x_):
    residual = x - x_
    out = np.zeros(len(residual))
    # for res in residual:
    #     print(res)

    for i in range(0,len(residual)):
        # 对每一个patch求重构残差
        temp = residual[i,:,:,:]
        temp1 = temp**2
        temp2 = temp1.sum().cpu()
        out[i] = np.sqrt(temp2)

    return out



def cal_residual(model_name, layer, gamma=2):
    model_path = os.path.join(pyramid_path,model_name)
    images_path = os.path.join(pyramid_path,layer)
    images_salted_path = os.path.join(pyramid_path,layer+'_salted')

    # 声明模型结构
    model = CDAE()

    # 将模型设置为评估模式
    model.eval()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # 加载保存的模型参数
    model.load_state_dict(torch.load(model_path))
    if device.type == 'cuda':
        model.cuda()
    
    total_res = []
    
    files = os.listdir(images_path)
    count = 0
    for image in files:
        # 加载输入数据
        # input_data = cv2.imread(os.path.join(images_salted_path,image))

        # 加载原始数据
        input_image = cv2.imread(os.path.join(images_path,image)).astype('float32')/255
        

        #划分patch
        patches = patchify(input_image,(8,8,3),step=4)
        patches =  torch.tensor(patches[:,:,0,:,:].reshape(patches.shape[0]*patches.shape[1],8,8,3)).to(device)

        input_data = patches.transpose(1, 3)

        #label_data = torch.tensor(patch(label_data)).transpose(1, 3).to(device)
        #print(input_data.shape,label_data.shape)

        # 执行推理
        with torch.no_grad():
            # 将输入数据传递给模型进行降噪和重建
            output_data = model(input_data)


        # 计算残差图集
        res= residual(input_data, output_data)
        #print(residual.shape)
        #res = np.zeros((len(residual)))
        #print(res)


        # for i in range(0,len(residual)):
        #     # 对每一个patch求重构残差
        #     temp = residual[i,:,:,:]**2

        #     #p = torch.zeros(3)
        #     #for channel in  range(len(residual[i])):
        #         #p[channel] = temp[channel,:,:].sum()

        #     #res[i] = temp.sum()
        #     temp = np.sqrt(temp.cpu().sum())
        #     print(temp)
            

        total_res.append(res)



        #total_res.append(res)
        sys.stdout.write(f"inter:   {count}/{len(files)}   \r")
        sys.stdout.flush()
        count += 1

    t = torch.tensor(total_res)
    print(t.shape)
    t = t.reshape(t.shape[0]*t.shape[1])
    res = t.numpy()
    print(res.shape)
    # 求残差集的均值与标准差
    res_mean = res.mean()
    res_std = res.std()
    print(f'    mean: {res_mean},    std: {res_std}')

    # 确定分割阈值

    T = res_mean + gamma * res_std

    return T


if __name__ == '__main__':
    print('size_128:')
    T_128 = cal_residual('size_128.pt', 'layer3')
    print(f'T_128: {T_128}')

    print('size_256:')
    T_256 = cal_residual('size_256.pt', 'layer2')
    print(f'T_256: {T_256}')

    print('size_512:')
    T_512 = cal_residual('size_512.pt', 'layer1')
    print(f'T_512: {T_512}')
