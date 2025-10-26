import torch
import torch.nn as nn



class Double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 =CDW(out_ch)
        self.at11 = FEM(out_ch,out_ch)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.at11(x1)
        x3 = self.conv2(x2)+x2

        return x3

class In_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(In_conv, self).__init__()
        self.conv = Double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            Double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, Transpose=False):
        super(Up, self).__init__()

        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)
        else:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch//2, kernel_size=1),
                                    nn.LeakyReLU(inplace=True))
        self.conv = Double_conv(in_ch, out_ch)

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX//2,
                                    diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = torch.tanh(x)
        return x
###########################################################
class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=1, relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
###########################################################
class CDW(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)     # conv 1*1
        x = self.dwconv(x)  # dwconv
        x = self.act(x)     # GELU
        x = self.drop(x)
        x = self.fc2(x)    # conv 1*1
        x = self.drop(x)
        return x

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.inc = In_conv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512, Transpose=True)
        self.up2 = Up(512, 256, Transpose=True)
        self.up3 = Up(256, 128, Transpose=True)
        self.up4 = Up(128, 64, Transpose=True)

        self.outc = OutConv(64, 1)

    def forward(self, x0):
        # 计算掩码
        mask = self.get_mask(x0)
        i=x0
        x1 = self.inc(x0)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.down4(x4)

        d4 = self.up1(x5, x4)
        d3 = self.up2(d4, x3)
        d2 = self.up3(d3, x2)
        d1 = self.up4(d2, x1)

        x = self.outc(d1)
        x = x+i
        # 应用掩码
        y = x0 + mask * x
        return y

    def get_mask(self, x):
        sam_add = x.sum(dim=2) == 0
        mask_sam = torch.where(sam_add, 1, 0)
        mask_sam = mask_sam.unsqueeze(2)
        return mask_sam
