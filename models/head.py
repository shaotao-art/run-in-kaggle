import math
import torch.nn as nn
import torch
import torch.nn.functional as F
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
    


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelAtten(nn.Module):
    def __init__(self, num_channel, reduction=16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(num_channel, num_channel//reduction),
            nn.ReLU(),
            nn.Linear(num_channel//reduction, num_channel)
            )
    
    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d( x, (1, 1))
        avg_score = self.mlp( avg_pool )
        max_pool = F.adaptive_max_pool2d( x, (1, 1))
        max_score = self.mlp( max_pool )
        sum_score = avg_score + max_score
        return F.sigmoid(sum_score).unsqueeze(2).unsqueeze(3) * x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1 )

class SpitialAtten(nn.Module):
    def __init__(self):
        super(SpitialAtten, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Conv2d(2, 1, 7, 1, 3)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        # print(x_compress.shape, x_out.shape)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale
    
    
class ASF_Head(nn.Module):
    def __init__(self, in_channel, num_level, hidden_dim, num_class) -> None:
        super().__init__()
        self.channel_atten = ChannelAtten(512)
        self.spitial_atten = SpitialAtten()
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
        x = self.channel_atten(x)
        x = self.spitial_atten(x)
        out = self.out(x)
        return out