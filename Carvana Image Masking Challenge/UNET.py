import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

class DConv(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_chan, out_chan, kernel_size=3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_chan),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_chan = 3, out_chan = 1, features = [64,128,256,512]):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride = 2)
        
        for feature in features:
            self.downs.append(DConv(in_chan, feature))
            in_chan = feature
        
        for feature in features[::-1]:
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DConv(feature*2, feature))
        self.bottleneck = DConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_chan, 1)

    def forward(self, x):
        skip = []
        for down in self.downs:
            x = down(x)
            skip.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x)
        skip = skip[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_con = skip[idx//2]

            if x.shape != skip_con.shape:
                x = TF.resize(x, size=  skip_con.shape[2:])

            concat_skip = torch.cat([skip_con, x], dim = 1)
            x = self.ups[idx+1](concat_skip)
        return self.final_conv(x)