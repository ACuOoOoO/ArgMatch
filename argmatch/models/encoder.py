import torch
import torch.nn as nn
from argmatch.models.modules import *
class Encoder(nn.Module):
    def __init__(self, ):
        super(Encoder, self).__init__()
        # self.block1 = nn.Sequential(
        #                             BasicLayer( 6,  16, stride=1),
        #                             BasicLayer( 16,  24, stride=2),
        #                             BasicDWLayer( 24,  32),
        #                             BasicLayer( 32,  48, stride=1),
        #                             BasicDWLayer( 48,  32, relu=False))
        
        # self.skip1 = nn.Sequential( BasicLayer( 3,  8, stride=2),
        #                             BasicDWLayer( 8,  16),
        #                             BasicDWLayer( 16,  32, relu=False))
        self.block1 = nn.Sequential(
                                    BasicLayer( 3,  8, stride=1),
                                    nn.InstanceNorm2d(8),
                                    BasicLayer( 8,  16, stride=2),
                                    nn.InstanceNorm2d(16),
                                    BasicLayer( 16,  32, stride=1, relu=False, obn=False))
        
        self.skip1 = nn.Sequential( BasicLayer( 3,  8, stride=2),
                                    BasicDWLayer( 8,  16),
                                    BasicLayer( 16, 32),
                                    BasicDWLayer( 32,  32, relu=False))
        
        
        # self.input_norm = nn.InstanceNorm2d(3,eps=1e-3)
        self.block2 = DownBLK(32,96)
        self.block3 = DownBLK(64,192) #/8
        self.block4 = DownBLK(128,384) #/16
        self.fus_4 = fuser(96,64,128)
        self.fus_8 = fuser(144,128,160)
        
    def forward(self,x):
        x1 = self.skip1(x)+self.block1(x)
        #x1 = self.block1(x)
        # print(x1.abs().max())
        x2 = self.block2(x1)
        # print(x2.abs().max())
        x3 = self.block3(x2[:,:64]) #/8
        # print(x3.abs().max())
        x4 = self.block4(x3[:,:128]) #/16
        # print(x4.abs().max())
        x3 = self.fus_8(x3[:,-144:], x4[:,-128:])
        # print(x3.abs().max())
        x2 = self.fus_4(x2, x3[:,-64:])
        return x2, x3, x4