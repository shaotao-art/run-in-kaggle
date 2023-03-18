import torch
import torch.nn as nn
import torch.nn.functional as F
from models import resnet
from models.basic import Conv_BN_ReLU
from models import fpn
from models import head


class PAN(nn.Module):
    def __init__(self, backbone, neck, detection_head):
        super(PAN, self).__init__()
        self.backbone = resnet.__dict__[backbone['type']](**backbone['param'])

        in_channels = neck['in_channels']
        self.reduce_layer1 = Conv_BN_ReLU(in_channels[0], 128)
        self.reduce_layer2 = Conv_BN_ReLU(in_channels[1], 128)
        self.reduce_layer3 = Conv_BN_ReLU(in_channels[2], 128)
        self.reduce_layer4 = Conv_BN_ReLU(in_channels[3], 128)

        self.fpem1 = fpn.FPEM_v1(**neck)
        self.fpem2 = fpn.FPEM_v1(**neck)
        
        self.det_head = head.__dict__[detection_head['type']](**detection_head['param'])

    def _upsample(self, x, size, scale=1):
        _, _, H, W = size
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear')

    def forward(self, imgs):
        # backbone
        f = self.backbone(imgs)

        # reduce channel
        f1 = self.reduce_layer1(f[0])
        f2 = self.reduce_layer2(f[1])
        f3 = self.reduce_layer3(f[2])
        f4 = self.reduce_layer4(f[3])

        # FPEM
        f1_1, f2_1, f3_1, f4_1 = self.fpem1(f1, f2, f3, f4)
        f1_2, f2_2, f3_2, f4_2 = self.fpem2(f1_1, f2_1, f3_1, f4_1)

        # FFM
        f1 = f1_1 + f1_2
        f2 = f2_1 + f2_2
        f3 = f3_1 + f3_2
        f4 = f4_1 + f4_2
        f2 = self._upsample(f2, f1.size())
        f3 = self._upsample(f3, f1.size())
        f4 = self._upsample(f4, f1.size())
        f = torch.cat((f1, f2, f3, f4), 1)

        # detection
        det_out = self.det_head(f)
        
        det_out = self._upsample(det_out, imgs.size())
        return det_out