import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gumbel_softmax import gumbel_softmax

class Simple(nn.Module):
    def __init__(self):
        super(Simple, self).__init__()
        #self.dropout = nn.Dropout(0.99)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 4, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 75, 2, stride=2),
            nn.ReLU(True),
        )

    def forward(self, images, segs):
        #segs = self.dropout(segs)
        #print(images.shape, segs.shape)
        x = torch.cat((images, segs), 1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.linear = nn.Linear(75, 1)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        out = self.linear(x)
        return out.permute(0, 3, 1, 2)
    
class MaskingNet(nn.Module):
    def __init__(self):
        super(MaskingNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 4, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 4, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(4, 2, 2, stride=2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, epoch, step):
        x = self.encoder(x)
        x = self.decoder(x).permute(0, 2, 3, 1)
        return x
