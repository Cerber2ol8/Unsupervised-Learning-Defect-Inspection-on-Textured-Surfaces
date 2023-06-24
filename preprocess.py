import cv2
import os
import numpy as np
import pickle
import sys
root_path = 'C:/Users/cy/project/cv/data'
image_dir = os.path.join(root_path,'patches')
salted_dir = os.path.join(root_path,'patches_salted')

pyramid_path = os.path.join(root_path,"pyramid")

if not os.path.exists(pyramid_path):
    os.mkdir(pyramid_path)

img_dir_layer1 = os.path.join(pyramid_path,"layer1")
if not os.path.exists(img_dir_layer1):
    os.mkdir(img_dir_layer1)
    
img_dir_layer2 = os.path.join(pyramid_path,"layer2")
if not os.path.exists(img_dir_layer2):
    os.mkdir(img_dir_layer2)

img_dir_layer3 = os.path.join(pyramid_path,"layer3")
if not os.path.exists(img_dir_layer3):
    os.mkdir(img_dir_layer3)

img_dir_layer1_salted = os.path.join(pyramid_path,"layer1_salted")
if not os.path.exists(img_dir_layer1_salted):
    os.mkdir(img_dir_layer1_salted)
    
img_dir_layer2_salted = os.path.join(pyramid_path,"layer2_salted")
if not os.path.exists(img_dir_layer2_salted):
    os.mkdir(img_dir_layer2_salted)

img_dir_layer3_salted = os.path.join(pyramid_path,"layer3_salted")
if not os.path.exists(img_dir_layer3_salted):
    os.mkdir(img_dir_layer3_salted)

def pyramid():

    # 进行金字塔
    for image_file in os.listdir(image_dir):
        path = os.path.join(salted_dir, image_file)
        image = cv2.imread(path)
        #print(image.shape)
        img_512 = image #512*512
        cv2.imwrite(os.path.join(img_dir_layer1,image_file),img_512)
        img_256 = cv2.pyrDown(img_512)
        cv2.imwrite(os.path.join(img_dir_layer2,image_file),img_256)
        img_128 = cv2.pyrDown(img_256)
        cv2.imwrite(os.path.join(img_dir_layer3,image_file),img_128)
        #img_64 = cv2.pyrDown(img_128)

    # 进行金字塔池化
    for image_file in os.listdir(salted_dir):
        path = os.path.join(salted_dir, image_file)
        image = cv2.imread(path)
        #print(image.shape)
        img_512 = image #512*512
        cv2.imwrite(os.path.join(img_dir_layer1_salted,image_file),img_512)
        img_256 = cv2.pyrDown(img_512)
        cv2.imwrite(os.path.join(img_dir_layer2_salted,image_file),img_256)
        img_128 = cv2.pyrDown(img_256)
        cv2.imwrite(os.path.join(img_dir_layer3_salted,image_file),img_128)
        #img_64 = cv2.pyrDown(img_128)


def cropImg(size,stride=4,patch_size=8):
    assert stride<=patch_size
    # 裁切图片

    m = int(size/4) - 1
    path = {512:img_dir_layer1,256:img_dir_layer2,128:img_dir_layer3}
    img_dir = path[size]
    filelist = os.listdir(img_dir)

    # 裁剪之后的图像矩阵
    img_crop = np.zeros((m*m*len(filelist),patch_size,patch_size,3),np.uint8)
    img_crop_salted = np.zeros((m*m*len(filelist),patch_size,patch_size,3),np.uint8)
    data = np.zeros((2,m*m*len(filelist),patch_size,patch_size,3),np.uint8)

    count = 0
    for i in range(len(filelist)):
        img = cv2.imread(os.path.join(img_dir,filelist[i]))
        img_salted = cv2.imread(os.path.join(img_dir+'_salted',filelist[i]))

        for j in range(0,m):
            for k in range(0,m):
                #print(img_crop.shape,img.shape)
                img_crop[count,:,:,:] = img[stride*j:stride*j+patch_size,stride*k:stride*k+patch_size,:]
                img_crop_salted[count,:,:,:] = img_salted[stride*j:stride*j+patch_size,stride*k:stride*k+patch_size,:]
                count += 1
                sys.stdout.write(f"img_{size} inter:{count} / {m*m*len(filelist)}\r")
                sys.stdout.flush()
    data[0] = img_crop
    data[1] = img_crop_salted
    print(f'crop {size} done!\r') 
    return data






def shuffle(data,n=10):
    # 多次打乱顺序，使之完全shuffled
    for i in range(0,n):
        np.random.shuffle(data)
        sys.stdout.write(f"shuffled:{i} / {n}\r")
        sys.stdout.flush()
    return data

def save(data, file):
    # 将裁剪后的图片矩阵保存
    out = open(os.path.join(pyramid_path, file),'wb')
    pickle.dump(data,out)
    out.close()
    print(f'save {file} done!\r')




if __name__ == '__main__':
    pyramid()
    for size in [512,256,128]:
        data = cropImg(size)
        data = shuffle(data)
        save(data,f'size_{size}.pkl')

        pass