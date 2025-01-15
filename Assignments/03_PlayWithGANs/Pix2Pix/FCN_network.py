import torch.nn as nn
import torch

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)、
         #(256-4+2*1)/2 + 1 = 128 [1, 3, 256, 256]->[1, 8, 128, 128]        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )

        ### FILL: add more CONV Layers
        # (128-4+2*1)/2 + 1 = 64 [1, 16, 64, 64]
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )

        # (64-4+2*1)/2 + 1 = 32 [1, 32, 32, 32]
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        #(32-4+2*1)/2 + 1 = 16   [1, 64, 16, 16]
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )

        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )

        self.deconv6 = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 3
            nn.BatchNorm2d(3),
            # nn.ReLU(inplace=True)
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)  
        x2 = self.conv2(x1)  
        x3 = self.conv3(x2)        
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # Decoder forward pass
        x6 = self.deconv1(x6)
        x6 = torch.cat([x6, x5], dim=1)
        x6 = self.deconv2(x6)
        x6 = torch.cat([x6, x4], dim=1)

        x6 = self.deconv3(x6)
        x6 = torch.cat([x6, x3], dim=1)

        x6 = self.deconv4(x6)
        x6 = torch.cat([x6, x2], dim=1)

        x6 = self.deconv5(x6)
        x6 = torch.cat([x6, x1], dim=1)
        ### FILL: encoder-decoder forward pass

        output = self.deconv6(x6)
        
        return output
    