import cv2
from CDAE import CDAE
import torch
from utils import pyramid,cut_img
import numpy as np
import os
from patchify import patchify

T_256 = 1.0828653288412886
T_128 = 1.6149606949554367
T_512 = 1.3385656836664561


def residual(x,x_):
    residual = x - x_
    out = np.zeros(len(residual))
    for i in range(0,len(residual)):
        # 对每一个patch求重构残差
        temp = residual[i,:,:,:]**2
        temp = temp.sum().cpu()
        out[i] = np.sqrt(temp)

    return out




def infer(model_path, image, T):


    # 声明模型结构
    model = CDAE()

    # 将模型设置为评估模式
    model.eval()
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载保存的模型参数
    model.load_state_dict(torch.load(model_path))
    if device.type == 'cuda':
        model.cuda()

    patches = patchify(image,(8,8,3),step=1)
    patches = patches[:,:,0,:,:]

    inputs = patches.reshape(patches.shape[0]*patches.shape[1],8,8,3)
    inputs = torch.tensor(inputs).transpose(1, 3).to(device)
    #print(inputs.shape)
    img_defect_seg = torch.zeros(patches.shape[0],patches.shape[1])

    outputs = []




    # 执行推理 使用batch输入模型，防止直接输入爆显存
    with torch.no_grad():

        batch_size = 1024 # 定义批次大小

        # 按照指定的批次大小拆分输入张量
        input_batches = torch.split(inputs, batch_size)

        # 逐个处理每个批次
        for i, batch in enumerate(input_batches):
            # 将输入数据传递给模型进行降噪和重建
            output_data = model(batch)
            res = residual(batch,output_data)
            for data in res:
                outputs.append(data)



    #res = residual(inputs.cpu(),output_data.cpu())
    outputs = np.array(outputs)
    print(outputs.shape)
    outputs = np.array(outputs).reshape(patches.shape[0],patches.shape[1])
    print(outputs.shape)

    for px in range(outputs.shape[0]):
        for py in range(outputs.shape[1]):
            if outputs[px,py] > T:
                img_defect_seg[px,py] = 1
            else:
                img_defect_seg[px,py] = 0

    
    #print(img_defect_seg)

    img_defect_seg = img_defect_seg.numpy()
    #img_defect_seg = img_defect_seg*255
    return img_defect_seg








if __name__ == '__main__':
    root_path = 'C:/Users/cy/project/cv/data'

    models = ['size_512.pt','size_256.pt','size_128.pt']
    model_path = os.path.join(root_path,'pyramid')

    model_512 = os.path.join(model_path,models[0])
    model_256 = os.path.join(model_path,models[1])
    model_128 = os.path.join(model_path,models[2])

    image_path = os.path.join(root_path,'defect','defect991_p_14_5.jpg')

    # 获取512*512的图像
    # image = os.path.join(root_path,'val_image','defect991.bmp')
    # cut_img(image,os.path.join(root_path,'val'))

    # 图像金字塔
    img_512,img_256,img_128 = pyramid(cv2.imread(image_path).astype('float32')/255)
    #img_defect_seg_128 = infer(model_128,img_128,T=T_128)
    
    #print(img_defect_seg_128.shape)

    img_defect_seg_512 = infer(model_512,img_512,T=T_512)
    img_defect_seg_512 = cv2.resize(img_defect_seg_512,(512,512))
    R = cv2.pyrDown(cv2.pyrDown(img_defect_seg_512))


    img_defect_seg_256 = infer(model_256,img_256,T=T_256)
    img_defect_seg_256 = cv2.resize(img_defect_seg_256,(256,256))
    G = cv2.pyrDown(img_defect_seg_256)

    img_defect_seg_128 = infer(model_128,img_128,T=T_128)
    img_defect_seg_128 = cv2.resize(img_defect_seg_128,(128,128))
    B = img_defect_seg_128

    result_image = np.stack((R, G, B), axis=-1)
    cv2.imshow("Result",cv2.resize(result_image,(512,512)))
    # cv2.imshow("Result",cv2.resize(img_defect_seg,(512,512)))
    cv2.imshow("Raw",cv2.imread(image_path))

    cv2.waitKey(0)
    cv2.destroyAllWindows()




    pass
