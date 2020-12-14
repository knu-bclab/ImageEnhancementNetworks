import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_I(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv_O(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)        
    )

def conv_B_E(in_channel, hid_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, hid_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hid_channel, hid_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(hid_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv_B_D(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2, inplace=True),        
    )

def conv_S(in_channel, out_channel, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv_down(in_channel, out_channel):
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0),
        nn.LeakyReLU(0.2, inplace=True)
    )

def conv_up(in_channel, out_channel):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0),
        nn.LeakyReLU(0.2, inplace=True)
    )

class Generator(nn.Module):
    def __init__(self ,residualFactor):
        super(Generator, self).__init__()
        self.input =conv_I(3,  3,  1,  1, 0)    # 32x32

        self.L1 = conv_B_E(3,  16, 16, 3, 1, 1) # 32x32
        self.L2 = conv_B_E(16, 16, 16, 3, 1, 1) # 16x16
        self.L3 = conv_B_E(16, 16, 16, 3, 1, 1) # 8x8
        self.L4 = conv_B_E(16, 16, 16, 3, 1, 1) # 4x4
        self.L5 = conv_B_E(16, 16, 16, 3, 1, 1) # 2x2
        self.L6 = conv_B_E(16, 16, 16, 3, 1, 1) # 1x1

        self.Down12 = conv_down(16,16)        
        self.Down23 = conv_down(16,16)        
        self.Down34 = conv_down(16,16)        
        self.Down45 = conv_down(16,16)        
        self.Down56 = conv_down(16,16) 
              
        self.Up65 = conv_up(16,16)
        self.Up54 = conv_up(16,16)
        self.Up43 = conv_up(16,16)
        self.Up32 = conv_up(16,16)
        self.Up21 = conv_up(16,16)

        self.Down = nn.MaxPool2d(2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.L6d = conv_B_E(16, 16, 16, 3, 1, 1) # 1x1
        self.L5d = conv_B_E(32, 16, 16, 3, 1, 1) # 2x2
        self.L4d = conv_B_E(32, 16, 16, 3, 1, 1) # 4x4
        self.L3d = conv_B_E(32, 16, 16, 3, 1, 1) # 8x8
        self.L2d = conv_B_E(32, 16, 16, 3, 1, 1) # 16x16
        self.L1d = conv_B_E(32, 16, 16, 3, 1, 1) # 32x32

        self.output = conv_O(16, 3, 1, 1, 0)
        #using Residual ratio
        self.residualFactor = residualFactor

    def forward(self, img):
        img_i = self.input(img)      # 32x32x3
        out_L1 = self.L1(img_i)      # ->32x32x16
        
        out_L2 = self.Down(out_L1) # ->16x16        
        #out_L2 = self.Down12(out_L1) # ->16x16
        out_L2 = self.L2(out_L2)        

        out_L3 = self.Down(out_L2) # ->8x8
        #out_L3 = self.Down23(out_L2) # ->8x8
        out_L3 = self.L3(out_L3)

        out_L4 = self.Down(out_L3) # ->4x4
        #out_L4 = self.Down34(out_L3) # ->4x4
        out_L4 = self.L4(out_L4)

        out_L5 = self.Down(out_L4) # ->2x2
        #out_L5 = self.Down45(out_L4) # ->2x2
        out_L5 = self.L5(out_L5)

        out_L6 = self.Down(out_L5) # ->1x1
        #out_L6 = self.Down56(out_L5) # ->1x1
        out_L6 = self.L6(out_L6)
        
        #nonGrobalContext
        out_L6d = self.L6d(torch.cat([out_L6], dim=1)) # 1x1x32 -> 1x1x16
        out_L5d = self.Up65(out_L6d) # 1x1x16 -> 2x2x16
        #out_L5d = self.Up(out_L6d) # 1x1x16 -> 2x2x16        
        out_L5d = self.L5d(torch.cat([out_L5, out_L5d], dim=1)) # 2x2x48 -> 2x2x16
        out_L4d = self.Up54(out_L5d) # 2x2x16 -> 4x4x16
        #out_L4d = self.Up(out_L5d) # 2x2x16 -> 4x4x16        
        out_L4d = self.L4d(torch.cat([out_L4, out_L4d], dim=1)) # 4x4x48 -> 4x4x16
        out_L3d = self.Up43(out_L4d) # 4x4x16 -> 8x8x16
        #out_L3d = self.Up(out_L4d) # 4x4x16 -> 8x8x16
        out_L3d = self.L3d(torch.cat([out_L3, out_L3d], dim=1)) # 8x8x48 -> 8x8x16
        out_L2d = self.Up32(out_L3d) # 8x8x16 -> 16x16x16
        #out_L2d = self.Up(out_L3d) # 8x8x16 -> 16x16x16
        out_L2d = self.L2d(torch.cat([out_L2, out_L2d], dim=1)) # 16x16x48 -> 16x16x16
        out_L1d = self.Up21(out_L2d)  # 16x16x16 -> 32x32x16
        #out_L1d = self.Up(out_L2d)  # 16x16x16 -> 32x32x16

        output = self.output(out_L1d) # 32x32x16 -> 32x32x3
        
        #Residual
        return img + self.residualFactor*output
