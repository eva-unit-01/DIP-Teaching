import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from facades_dataset import FacadesDataset

device = 'cuda'

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )
        ### FILL: add more CONV Layers
        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        self.convt1 = nn.Sequential(
            

            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # Input channels: 32, Output channels: 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.convt2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # Input channels: 32, Output channels: 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.convt3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),  # Input channels: 32, Output channels: 16
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
        )

        self.convt4 = nn.Sequential(   
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),  # Input channels: 32, Output channels: 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
        )

        self.convt5 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
        )

            # nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 8
            # nn.BatchNorm2d(8),
            # nn.ReLU(inplace=True),

        self.convt6 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 3
            nn.BatchNorm2d(3),
            # nn.ReLU(inplace=True)
            nn.Tanh()
        )
        ### None: since last layer outputs RGB channels, may need specific activation function

    def forward(self, x):
        # Encoder forward pass
        
        # Decoder forward pass
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        x6 = self.convt1(x6)
        x6 = torch.cat([x6, x5], dim=1)

        x6 = self.convt2(x6)
        x6 = torch.cat([x6, x4], dim=1)

        x6 = self.convt3(x6)
        x6 = torch.cat([x6, x3], dim=1)

        x6 = self.convt4(x6)
        x6 = torch.cat([x6, x2], dim=1)

        x6 = self.convt5(x6)
        x6 = torch.cat([x6, x1], dim=1)

        # code = self.conv1(x)
        # output = self.conv2(code)
        output = self.convt6(x6)
        ### FILL: encoder-decoder forward pass

        # output = ...
        
        return output
    

class GanNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.generative = FullyConvNetwork()

        self.discriminative = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Dropout2d(),

            nn.Conv2d(256, 1, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        '''
        x.shape: (B, C, H, W)
        '''
        image = self.generative(x)

        return image

        

    def test(self):
        wzj = torch.randn(12, 3, 256, 256).to(device)
        print(self.discriminative(wzj).shape)

if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    # print(vars(train_loader))
    image, seman = next(iter(train_loader))
    image, seman = image.cuda(), seman.cuda()
    print(image.shape)
    test_net = GanNetwork().to(device)  
    # print(test_net(image).shape)
    gt = torch.cat([image, seman], dim=1)
    print(GanNetwork().cuda().discriminative(gt).shape)
    # test_net.test()
#     net = FullyConvNetwork()
    
#     input = torch.randn(1, 3, 256, 256)
#     output = net(input)
#     print(output.shape)