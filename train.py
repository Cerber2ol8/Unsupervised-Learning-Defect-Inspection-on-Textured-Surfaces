import matplotlib.pyplot as plt # plotting library
import numpy as np # this module is useful to work with numerical arrays
import pandas as pd 
import random 
import torch
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pickle
from CDAE import CDAE
import sys 
from datetime import datetime
from utils import CustomDataset

root_path = 'C:/Users/cy/project/cv/data'
image_dir = os.path.join(root_path,'train_image')
pyramid_path = os.path.join(root_path,"pyramid")

size_512 = os.path.join(pyramid_path,'size_512.pkl')
size_256 = os.path.join(pyramid_path,'size_256.pkl')
size_128 = os.path.join(pyramid_path,'size_128.pkl')
data_files = [size_512,size_256,size_128]







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





def train(model, dataloader, optimizer, criterion, info):
    model.train()
    total_loss = 0.0
    count = 1
    duration = 10
    start_time = datetime.now()



    for batch in dataloader:
        
        #print(len(dataloader))
        optimizer.zero_grad()
        # data包括原始patch和noised patch
        data = batch.to(device)
        # 这里输入 noise data作为input，原始patch作为label
        inputs = data[1].transpose(1, 3)
        labels = data[0].transpose(1, 3)


        #print(inputs.shape)
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # 计算重构均方误差

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if count % duration == 0:
            average_loss = total_loss / count
            end_time = datetime.now()
            time_diff = end_time - start_time
            left_time = (len(dataloader)-count)/duration * time_diff 
            total_seconds = left_time.total_seconds()
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            left_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            start_time = datetime.now()
            #print(f'inter:{count}/{len(dataloader)}')
            sys.stdout.write(f"{info} inter:{count}/{len(dataloader)} , Loss: {average_loss:.6f} , ETA: {left_time} \r")
            sys.stdout.flush()

            
        count += 1
        
    return total_loss / len(dataloader)



    
if __name__ == '__main__':

    task_id = 0
    size = [512,256,128]
    batch_size = [8192,8192,8192]
    num_epochs = [10,10,10]

    #需要进行三个训练任务
    for data_file in data_files:
        net = CDAE()
        net.to(device)
        
        optimizer = torch.optim.SGD(net.parameters(), lr=0.001,weight_decay=0.01)
        criterion = nn.MSELoss()
        dataset = CustomDataset(data_file)

        dataloader = DataLoader(dataset, batch_size=batch_size[task_id], shuffle=True)

        for epoch in range(num_epochs[task_id]):
            loss = train(net, dataloader, optimizer, criterion, f'Task {task_id} Epoch {epoch+1}/{num_epochs[task_id]},')
            print(f'Task {task_id} Epoch {epoch+1}/{num_epochs[task_id]},   Loss: {loss:.6f}')

        torch.save(net.state_dict(),f'{data_file[:-4]}.pt')
        task_id += 1


