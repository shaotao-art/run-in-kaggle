import math
import torch.nn as nn
import torch

__all__ = ['PA_Head', 'ASF_Head']

class PA_Head(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_classes):
        super(PA_Head, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,
                               hidden_dim,
                               kernel_size=3,
                               stride=1,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(hidden_dim,
                               num_classes,
                               kernel_size=1,
                               stride=1,
                               padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, f):
        out = self.conv1(f)
        out = self.relu1(self.bn1(out))
        out = self.conv2(out)

        return out
    


class SptialAtten(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()
        self.spatial_atten = nn.Sequential(*[
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, 3, 1, 1),
            nn.Sigmoid()
        ])
        self.out = nn.Sequential(*[
            nn.Conv2d(in_channel, 1, 3, 1, 1),
            nn.Sigmoid()
        ])


    def forward(self, x):
        input_ = x
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.spatial_atten(x)
        x = x + input_
        out = self.out(x)
        return out
    
    
class ASF_Head(nn.Module):
    def __init__(self, in_channel, num_level, hidden_dim, num_class) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channel, in_channel // num_level, 3, 1, 1)
        self.spatial_atten = SptialAtten(in_channel // 4)
        self.out = nn.Sequential(*[
            nn.Conv2d(in_channel, hidden_dim, 3, 1, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_class, 1, 1, 0)
        ])


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        input_ = x
        x = self.conv(x)
        x = input_ * self.spatial_atten(x)
        out = self.out(x)
        return out