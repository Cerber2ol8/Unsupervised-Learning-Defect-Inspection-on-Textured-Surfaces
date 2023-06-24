import numpy as np
from patchify import patchify

import cv2
import os

path = 'C:/Users/cy/project/cv/data'
raw_path = os.path.join(path, "raw")
patches_path = os.path.join(path, "patches")
salted_path = os.path.join(path, "patches_salted")

if not os.path.exists(patches_path):
    os.mkdir(patches_path)

if not os.path.exists(salted_path):
    os.mkdir(salted_path)

# 删除原来的图像
for file in os.listdir(patches_path):
    os.remove(os.path.join(patches_path, file))
for file in os.listdir(salted_path):
    os.remove(os.path.join(salted_path, file))


for file in os.listdir(raw_path):
    image_path = os.path.join(raw_path, file)

    image = cv2.imread(image_path)
    #image = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13, 14, 15, 16]])
    print(image.shape)
    patches = patchify(image, (512,512,3), step=512) 
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            patch = patches[i, j, 0, :, :, :]
            patch_file = os.path.join(patches_path,f'{file[:-4]}_p_{i}_{j}.jpg')
            salted_file = os.path.join(salted_path,f'{file[:-4]}_p_{i}_{j}.jpg')
            cv2.imwrite(patch_file, patch)
            cv2.imwrite(salted_file, patch)