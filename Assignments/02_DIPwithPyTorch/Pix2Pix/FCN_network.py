import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)、
         #(256-4+2*1)/2 + 1 = 128 [1, 3, 256, 256]->[1, 8, 128, 128]        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        ### FILL: add more CONV Layers
        # (128-4+2*1)/2 + 1 = 64 [1, 16, 64, 64]
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # (64-4+2*1)/2 + 1 = 32 [1, 32, 32, 32]
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        #(32-4+2*1)/2 + 1 = 16   [1, 64, 16, 16]
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        ### None: since last layer outputs RGB channels, may need specific activation function
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # 论文中提到的激活层，避免负值
        )


    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)  
        x = self.conv2(x)  
        x = self.conv3(x)        
        x = self.conv4(x)
        # Decoder forward pass
        x = self.deconv1(x)  
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)  
        ### FILL: encoder-decoder forward pass

        output = x  
        
        return output
    