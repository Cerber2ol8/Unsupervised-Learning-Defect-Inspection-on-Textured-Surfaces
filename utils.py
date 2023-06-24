import cv2
import math 
import numpy as np
from torch.utils.data import Dataset
import torch
import pickle


def Illumination_Normalization(img,method = 'hist'):
    '''
    the function to carry out the illumination normalization using Weber Local Descriptor or Histogram Equalization

    '''
    height,width,channel = np.shape(img)
    
    if method == 'hist':
        # 分通道做直方图均衡
        equ0 = cv2.equalizeHist(img[:,:,0])
        equ1 = cv2.equalizeHist(img[:,:,1])
        equ2 = cv2.equalizeHist(img[:,:,2])
        img_hist_equ = np.zeros((height,width,channel),np.uint8)
        img_hist_equ[:,:,0] = equ0
        img_hist_equ[:,:,1] = equ1
        img_hist_equ[:,:,2] = equ2
        return img_hist_equ
    
    if method == 'wld':
        # 外部补零
        img_padding = np.zeros((height+2,width+2,channel),np.float32)
        img_padding[1:height+1,1:width+1,:] = img.astype('float32')
        
        temp = np.zeros((height+2,width+2,channel),np.float32)
        
        for k in range(0,channel):
            for i in range(1,height+1):
                for j in range(1,width+1):
                    temp[i,j,k] = math.atan(9- 1/img_padding[i,j,k]*(img_padding[i-1,j-1,k] + img_padding[i,j-1,k]  + img_padding[i+1,j-1,k]
                                                                    +img_padding[i-1,j,k]   + img_padding[i,j,k]    + img_padding[i+1,j,k]
                                                                    +img_padding[i-1,j+1,k] + img_padding[i,j+1,k]  + img_padding[i+1,j+1,k]))
                    temp[i,j,k] = temp[i,j,k]*180/math.pi

        return temp[1:height+1,1:width+1,:]
    
# 两种椒盐噪声函数

def saltpepper(img,n):
    m=int((img.shape[0]*img.shape[1])*n)
    for a in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=255
        elif img.ndim==3:
            img[j,i,0]=255
            img[j,i,1]=255
            img[j,i,2]=255
    for b in range(m):
        i=int(np.random.random()*img.shape[1])
        j=int(np.random.random()*img.shape[0])
        if img.ndim==2:
            img[j,i]=0
        elif img.ndim==3:
            img[j,i,0]=0
            img[j,i,1]=0
            img[j,i,2]=0
    return img

def salt_and_pepper(img,p):
    temp = img
    channel = img.ndim
    # 灰度图
    if channel == 2:
        height,width = np.shape(img)[0:2]
        for i in range(0,height):
            for j in range(0,width):
                if np.random.random()<p:# 噪声强度
                    if np.random.random() < 0.5:
                        temp[i,j] = 255 # salt
                    if np.random.random() > 0.5:
                        temp[i,j] = 0 # pepper
    # 彩图
    if channel == 3:
        height,width = np.shape(img)[0:2]
        for i in range(0,height):
            for j in range(0,width):
                if np.random.random()<p:# 噪声强度
                    if np.random.random() < 0.5:
                        temp[i,j,0] = 255 # salt
                        temp[i,j,1] = 255
                        temp[i,j,2] = 255
                    if np.random.random() > 0.5:
                        temp[i,j,0] = 0 # pepper
                        temp[i,j,1] = 0
                        temp[i,j,2] = 0
    return temp


class CustomDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'rb') as file:
            loaded_data = pickle.load(file)
        self.data = np.transpose(loaded_data,(1, 0, 2, 3, 4))
        print(self.data.shape)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index].astype('float32')
        sample /= 255
        return torch.from_numpy(sample)
    


def pyramid(image, size=512):
    assert image.shape[0] == size
    if size != 512:
        image = cv2.resize(image,(512,512))
    img_512 = image #512*512
    img_256 = cv2.pyrDown(img_512)
    img_128 = cv2.pyrDown(img_256)

    return img_512,img_256,img_128


def cut_img(src_img, dst_path, size=512):
    # 将原始图像裁剪成固定大小的图像

    import os
    from patchify import patchify
    assert os.path.exists(src_img)
    img = cv2.imread(src_img)

    patches = patchify(img, (size,size,3), step=size) 
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0, :, :, :]
            patch_file = os.path.join(dst_path,f'{os.path.basename(src_img)[:-4]}_p_{i}_{j}.jpg')
            cv2.imwrite(patch_file, patch)
    