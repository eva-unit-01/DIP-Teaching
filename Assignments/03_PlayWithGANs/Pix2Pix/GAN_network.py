import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import FullyConvNetwork
class GanNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = FullyConvNetwork()
        self.discriminator = nn.Sequential(
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
        
        image = self.generator(x)
        return image