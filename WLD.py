import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

image = cv2.imread("img.jpg")

# 将图像转换为灰度图
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def WLD(x):
        kernel = torch.Tensor([[[[-1, -1, -1],
                                   [-1, -1, -1],
                                   [-1, -1, -1]]]])

        kernel_1 = torch.Tensor([[[[-1, -1, 1],
                                   [-1, 0, -1],
                                   [1, -1, -1]]]])

        conv = torch.nn.Conv2d(1, 1, (3, 3), stride=1, padding=1, bias=False)
        conv.weight.data = kernel

        #weight = nn.Parameter(data=kernel, requires_grad=False)
        x = x.unsqueeze(0)
        

        x = (9*x + conv(x))/x
        x = torch.arctan(x)
        print(x[0])

        return x
    

output = WLD(torch.Tensor(image))


print(output.shape)




# 将NumPy数组转换为OpenCV图像（BGR格式）
image = cv2.cvtColor(output.detach().numpy().transpose(1, 2, 0), cv2.COLOR_RGB2BGR)


print(image.shape)

cv2.imshow("",image)
cv2.waitKey(0)
cv2.destroyAllWindows()