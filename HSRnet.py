import torch
import torch.nn as nn
from math import sqrt


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class HSI_ReNet_g(nn.Module):
    def __init__(self,in_channels,out_channel):
        super(HSI_ReNet_g, self).__init__()
        self.conv_B = self.convlayer(5, 6, 3)
        self.conv_G = self.convlayer(5, 10, 3)
        self.conv_R = self.convlayer(5, 15, 3)
        self.convinputb = self.convlayer(6, 6, 1)
        self.convinputg = self.convlayer(10, 10, 1)
        self.convinputr = self.convlayer(15, 15, 1)
        self.conv1 = self.convlayer(out_channel, out_channel, 3)
        self.e1 = ChannelAttention(out_channel)
        self.HSI1 = HSI_block(out_channel, out_channel)
        self.en1 = ChannelAttention(out_channel)
        self.conv2 = self.convlayer(out_channel, out_channel, 3)
        self.e2 = ChannelAttention(out_channel)
        self.HSI2 = HSI_block(out_channel, out_channel)
        self.en2 = ChannelAttention(out_channel)
        self.conv3 = self.convlayer(out_channel, out_channel, 3)
        self.e3 = ChannelAttention(out_channel)
        self.HSI3 = HSI_block(out_channel, out_channel)
        self.en3 = ChannelAttention(out_channel)
        self.conv4 = self.convlayer(out_channel, out_channel, 3)
        self.e4 = ChannelAttention(out_channel)
        self.HSI4 = HSI_block(out_channel, out_channel)
        self.en4 = ChannelAttention(out_channel)
        self.conv5 = self.convlayer(out_channel, out_channel, 3)
        self.e5 = ChannelAttention(out_channel)
        self.HSI5 = HSI_block(out_channel, out_channel)
        self.en5 = ChannelAttention(out_channel)
        self.conv6 = self.convlayer(out_channel, out_channel, 3)
        self.e6 = ChannelAttention(out_channel)
        self.HSI6 = HSI_block(out_channel, out_channel)
        self.en6 = ChannelAttention(out_channel)
        self.conv7 = self.convlayer(out_channel, out_channel, 3)
        self.e7 = ChannelAttention(out_channel)
        self.HSI7 = HSI_block(out_channel, out_channel)
        self.en7 = ChannelAttention(out_channel)
        self.conv8 = self.convlayer(out_channel, out_channel, 3)
        self.e8 = ChannelAttention(out_channel)
        self.HSI8 = HSI_block(out_channel, out_channel)
        self.en8 = ChannelAttention(out_channel)
        self.conv9 = self.convlayer(out_channel, out_channel, 3)
        self.e9 = ChannelAttention(out_channel)
        self.HSI9 = HSI_block(out_channel, out_channel)
        self.en9 = ChannelAttention(out_channel)
        self.convf = self.convlayer(2 * out_channel, out_channel, 1)



        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        Lrelu = nn.LeakyReLU(0.2, inplace=True)
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        (batch,_,h,w)=x.shape
        input_R=torch.randn((batch,1,h,w))
        input_G=torch.randn((batch,1,h,w))
        input_B=torch.randn((batch,1,h,w))
        if x.device != torch.device('cpu'):
            input_R=input_R.cuda()
            input_G=input_G.cuda()
            input_B=input_B.cuda()
        input_R[:,0,:,:] = x[:, 0, :, :]
        input_G[:,0,:,:] = x[:, 1, :, :]
        input_B[:,0,:,:] = x[:, 2, :, :]
        input_RG = input_R - input_G
        input_GB = input_G - input_B
        input = torch.cat([input_R, input_RG], 1)
        input = torch.cat([input, input_G], 1)
        input = torch.cat([input, input_GB], 1)
        input = torch.cat([input, input_B], 1)
        B = self.conv_B(input)
        G = self.conv_G(input)
        R = self.conv_R(input)
        B = self.convinputb(B)
        G = self.convinputg(G)
        R = self.convinputr(R)
        f0 = torch.cat([B, G], 1)
        f0 = torch.cat([f0, R], 1)
        Sf0 = self.HSI1(f0)
        e1 = self.e1(f0)
        en1 = self.en1(f0)
        f0_e = f0 * e1
        Sf0_en = Sf0 * en1
        f0_conv = self.conv1(f0)
        f1 = torch.add(f0_conv, f0_e)
        f1 = torch.add(f1, Sf0_en)
        Sf1 = self.HSI2(f1)
        e2 = self.e2(f1)
        en2 = self.en2(f1)
        f1_e = f1 * e2
        Sf1_en = Sf1 * en2
        f1_conv = self.conv2(f1)
        f2 = torch.add(f1_conv, f1_e)
        f2 = torch.add(f2, Sf1_en)
        Sf2 = self.HSI3(f2)
        e3 = self.e3(f2)
        en3 = self.en3(f2)
        f2_e = f2 * e3
        Sf2_en = Sf2 * en3
        f2_conv = self.conv3(f2)
        f3 = torch.add(f2_conv, f2_e)
        f3 = torch.add(f3, Sf2_en)
        Sf3 = self.HSI4(f3)
        e4 = self.e4(f3)
        en4 = self.en4(f3)
        f3_e = f3 * e4
        Sf3_en = Sf3 * en4
        f3_conv = self.conv4(f3)
        f4 = torch.add(f3_conv, f3_e)
        f4 = torch.add(f4, Sf3_en)
        Sf4 = self.HSI5(f4)
        e5 = self.e5(f4)
        en5 = self.en5(f4)
        f4_e = f4 * e5
        Sf4_en = Sf4 * en5
        f4_conv = self.conv5(f4)
        f5 = torch.add(f4_conv, f4_e)
        f5 = torch.add(f5, Sf4_en)
        Sf5 = self.HSI6(f5)
        e6 = self.e6(f5)
        en6 = self.en6(f5)
        f5_e = f5 * e6
        Sf5_en = Sf5 * en6
        f5_conv = self.conv6(f5)
        f6 = torch.add(f5_conv, f5_e)
        f6 = torch.add(f6, Sf5_en)
        Sf6 = self.HSI7(f6)
        e7 = self.e7(f6)
        en7 = self.en7(f6)
        f6_e = f6 * e7
        Sf6_en = Sf6 * en7
        f6_conv = self.conv7(f6)
        f7 = torch.add(f6_conv, f6_e)
        f7 = torch.add(f7, Sf6_en)
        Sf7 = self.HSI8(f7)
        e8 = self.e8(f7)
        en8 = self.en8(f7)
        f7_e = f7 * e8
        Sf7_en = Sf7 * en8
        f7_conv = self.conv8(f7)
        f8 = torch.add(f7_conv, f7_e)
        f8 = torch.add(f8, Sf7_en)
        Sf8 = self.HSI9(f8)
        e9 = self.e9(f8)
        en9 = self.en9(f8)
        f8_e = f8 * e9
        Sf8_en = Sf8 * en9
        f8_conv = self.conv9(f8)
        f9 = torch.add(f8_conv, f8_e)
        f9 = torch.add(f9, Sf8_en)
        f9 = torch.add(f0, f9)
        f9 = torch.cat([f0, f9], 1)
        f9 = self.convf(f9)
        self.feature0 = f0
        self.feature1 = f1
        self.feature2 = f2
        self.feature3 = f3
        self.feature4 = f4
        self.feature5 = f5
        self.feature6 = f6
        self.feature7 = f7
        self.feature8 = f8
        self.feature9 = f9
        return f9

class HSI_block(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(HSI_block, self).__init__()
        self.conv1 = self.convlayer(in_channels,128,3)
        self.conv2 = nn.Conv2d(128, out_channels, 3, stride=1,padding=1, bias=True)
        self.conv3 = self.convlayer(out_channels,out_channels,1)
        self.Lrelu = nn.LeakyReLU(0.2, inplace=True)

    def convlayer(self, in_channels, out_channels, kernel_size):
        pader = nn.ReplicationPad2d(int((kernel_size - 1) / 2))
        conver = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True)
        Lrelu = nn.LeakyReLU(0.2, inplace=True)
        layers = filter(lambda x: x is not None, [pader, conver, Lrelu])
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        residuals=torch.add(x,out2)
        residuals=self.Lrelu(residuals)
        output=self.conv3(residuals)
        return output