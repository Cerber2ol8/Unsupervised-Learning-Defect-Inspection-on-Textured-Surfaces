import torch
import torch.nn as nn

class CDAE(nn.Module):
    def __init__(self):
        super(CDAE, self).__init__()

        # 编码器层
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),   # 第一层卷积层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # 第二层卷积层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

                
        # 解码器层
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), # 第一层反卷积层
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 第一层反卷积层
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),    # 第二层反卷积层
            nn.Sigmoid()
        )

        
        # self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        # self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        # self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv6 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.conv7 = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.relu(x)
        # x = self.maxpool2(x)
        # x = self.conv5(x)
        # x = self.relu(x)
        # x = self.upsample2(x)
        # x = self.conv6(x)
        # x = self.relu(x)
        # x = self.upsample3(x)
        # x = self.conv7(x)
        # x = self.sigmoid(x)
        #print(x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

if __name__ == '__main__':
    # 创建模型实例
    model = CDAE()

    # 打印模型的摘要信息
    print(model)
    input_shape = model.encoder.in_features
    print("Input:",input_shape)