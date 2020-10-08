import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class BasicBlock(nn.Module):  #basic block for Conv2d
    def __init__(self,in_planes,planes,stride=1):
        super(BasicBlock,self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(planes)
        self.shortcut=nn.Sequential()
    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return out

class ThreeDConv(nn.Module):
    def __init__(self,in_planes,planes,stride=1):
        super(ThreeDConv, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3=nn.Conv3d(planes,planes,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm3d(planes)

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=F.relu(self.bn2(self.conv2(out)))
        out=F.relu(self.bn3(self.conv3(out)))
        return out

class GC_NET(nn.Module):
    def __init__(self,block,block_3d,num_block,height,width,maxdisp):
        super(GC_NET, self).__init__()
        self.height = height
        self.width = width
        self.maxdisp = int(maxdisp/2)
        self.in_planes = 32
        # first two conv2d
        self.conv0 = nn.Conv2d(3, 32, 5, 2, 2)
        self.bn0 = nn.BatchNorm2d(32)
        # res block
        self.res_block = self._make_layer(block, self.in_planes, 32, num_block[0], stride=1)
        # last conv2d
        self.conv1 = nn.Conv2d(32, 32, 3, 1, 1)

        # conv3d
        self.conv3d_1 = nn.Conv3d(64, 32, 3, 1, 1)
        self.bn3d_1 = nn.BatchNorm3d(32)
        self.conv3d_2 = nn.Conv3d(32, 32, 3, 1, 1)
        self.bn3d_2 = nn.BatchNorm3d(32)

        self.conv3d_3 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_3 = nn.BatchNorm3d(64)
        self.conv3d_4 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_4 = nn.BatchNorm3d(64)
        self.conv3d_5 = nn.Conv3d(64, 64, 3, 2, 1)
        self.bn3d_5 = nn.BatchNorm3d(64)

        # conv3d sub_sample block
        self.block_3d_1 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_2 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_3 = self._make_layer(block_3d, 64, 64, num_block[1], stride=2)
        self.block_3d_4 = self._make_layer(block_3d, 64, 128, num_block[1], stride=2)

        # deconv3d
        self.deconv1 = nn.ConvTranspose3d(128, 64, 3, 2, 1, 1)
        self.debn1 = nn.BatchNorm3d(64)
        self.deconv2 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn2 = nn.BatchNorm3d(64)
        self.deconv3 = nn.ConvTranspose3d(64, 64, 3, 2, 1, 1)
        self.debn3 = nn.BatchNorm3d(64)
        self.deconv4 = nn.ConvTranspose3d(64, 32, 3, 2, 1, 1)
        self.debn4 = nn.BatchNorm3d(32)

        # last deconv3d
        self.deconv5 = nn.ConvTranspose3d(32, 1, 3, 2, 1, 1)

        self.regression = DisparityRegression(maxdisp)

    def forward(self, imgLeft, imgRight):
        original_size = [1, self.maxdisp*2, imgLeft.size(2), imgLeft.size(3)]
        imgl0 = F.relu(self.bn0(self.conv0(imgLeft)))
        imgr0 = F.relu(self.bn0(self.conv0(imgRight)))

        imgl_block = self.res_block(imgl0)
        imgr_block = self.res_block(imgr0)

        imgl1 = self.conv1(imgl_block)
        imgr1 = self.conv1(imgr_block)
                # cost volume
        cost_volum = self.cost_volume(imgl1, imgr1)
            # print(cost_volum.shape)
        conv3d_out = F.relu(self.bn3d_1(self.conv3d_1(cost_volum)))
        conv3d_out = F.relu(self.bn3d_2(self.conv3d_2(conv3d_out)))
        # conv3d block
        conv3d_block_1 = self.block_3d_1(cost_volum)
        conv3d_21 = F.relu(self.bn3d_3(self.conv3d_3(cost_volum)))
        conv3d_block_2 = self.block_3d_2(conv3d_21)
        conv3d_24 = F.relu(self.bn3d_4(self.conv3d_4(conv3d_21)))
        conv3d_block_3 = self.block_3d_3(conv3d_24)
        conv3d_27 = F.relu(self.bn3d_5(self.conv3d_5(conv3d_24)))
        conv3d_block_4 = self.block_3d_4(conv3d_27)

        # deconv
        deconv3d = F.relu(self.debn1(self.deconv1(conv3d_block_4)) + conv3d_block_3)
        deconv3d = F.relu(self.debn2(self.deconv2(deconv3d)) + conv3d_block_2)
        deconv3d = F.relu(self.debn3(self.deconv3(deconv3d)) + conv3d_block_1)
        deconv3d = F.relu(self.debn4(self.deconv4(deconv3d)) + conv3d_out)

        # last deconv3d
        deconv3d = self.deconv5(deconv3d)
        out = deconv3d.view( original_size)
        prob = F.softmax(-out, 1)


        disp1 = self.regression(prob)

        return disp1



    def _make_layer(self,block,in_planes,planes,num_block,stride):
        strides=[stride]+[1]*(num_block-1)
        layers=[]
        for step in strides:
            layers.append(block(in_planes,planes,step))
        return nn.Sequential(*layers)

    def cost_volume(self,imgl,imgr):
        B, C, H, W = imgl.size()
        cost_vol = torch.zeros(B, C * 2, self.maxdisp , H, W).type_as(imgl)
        for i in range(self.maxdisp):
            if i > 0:
                cost_vol[:, :C, i, :, i:] = imgl[:, :, :, i:]
                cost_vol[:, C:, i, :, i:] = imgr[:, :, :, :-i]
            else:
                cost_vol[:, :C, i, :, :] = imgl
                cost_vol[:, C:, i, :, :] = imgr
        return cost_vol

    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

def GcNet(height, width, maxdisp):
    return GC_NET(BasicBlock, ThreeDConv, [8, 1], height, width, maxdisp)
    # return GC_NET_new(BasicBlock, [8, 1], height, width, maxdisp)

class DisparityRegression(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.disp_score = torch.range(0, max_disp - 1)  # [D]
        self.disp_score = self.disp_score.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # [1, D, 1, 1]


    def forward(self, prob):
        disp_score = self.disp_score.expand_as(prob).type_as(prob)  # [B, D, H, W]
        out = torch.sum(disp_score * prob, dim=1)  # [B, H, W]
        return out


class PSMNet(nn.Module):

    def __init__(self, max_disp):
        super().__init__()

        self.cost_net = CostNet()
        self.stackedhourglass = StackedHourglass(max_disp)
        self.D = max_disp

        self.__init_params()

    def forward(self, left_img, right_img):
        original_size = [self.D, left_img.size(2), left_img.size(3)]

        left_cost = self.cost_net(left_img)  # [B, 32, 1/4H, 1/4W]
        right_cost = self.cost_net(right_img)  # [B, 32, 1/4H, 1/4W]
        # cost = torch.cat([left_cost, right_cost], dim=1)  # [B, 64, 1/4H, 1/4W]
        # B, C, H, W = cost.size()

        # print('left_cost')
        # print(left_cost[0, 0, :3, :3])

        B, C, H, W = left_cost.size()

        cost_volume = torch.zeros(B, C * 2, self.D // 4, H, W).type_as(left_cost)  # [B, 64, D, 1/4H, 1/4W]

        # for i in range(self.D // 4):
        #     cost_volume[:, :, i, :, i:] = cost[:, :, :, i:]

        for i in range(self.D // 4):
            if i > 0:
                cost_volume[:, :C, i, :, i:] = left_cost[:, :, :, i:]
                cost_volume[:, C:, i, :, i:] = right_cost[:, :, :, :-i]
            else:
                cost_volume[:, :C, i, :, :] = left_cost
                cost_volume[:, C:, i, :, :] = right_cost

        disp1, disp2, disp3 = self.stackedhourglass(cost_volume, out_size=original_size)

        return disp1, disp2, disp3

    def __init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

class CostNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.cnn = CNN()
        self.spp = SPP()
        self.fusion = nn.Sequential(
                Conv2dBn(in_channels=288, out_channels=128, kernel_size=3, stride=1, padding=1, use_relu=True),
                nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
            )

    def forward(self, inputs):
        conv2_out, conv4_out = self.cnn(inputs)           # [B, 64, 1/4H, 1/4W], [B, 128, 1/4H, 1/4W]

        spp_out = self.spp(conv4_out)                    # [B, 128, 1/4H, 1/4W]
        out = torch.cat([conv2_out, conv4_out, spp_out], dim=1)  # [B, 320, 1/4H, 1/4W]
        out = self.fusion(out)                            # [B, 32, 1/4H, 1/4W]

        return out


class SPP(nn.Module):

    def __init__(self):
        super().__init__()

        # self.branch1 = self.__make_branch(kernel_size=64, stride=64)
        self.branch2 = self.__make_branch(kernel_size=32, stride=32)
        self.branch3 = self.__make_branch(kernel_size=16, stride=16)
        self.branch4 = self.__make_branch(kernel_size=8, stride=8)

    def forward(self, inputs):

        out_size = inputs.size(2), inputs.size(3)
        # branch1_out = F.upsample(self.branch1(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        # print('branch1_out')
        # print(branch1_out[0, 0, :3, :3])
        branch2_out = F.upsample(self.branch2(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        branch3_out = F.upsample(self.branch3(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        branch4_out = F.upsample(self.branch4(inputs), size=out_size, mode='bilinear')  # [B, 32, 1/4H, 1/4W]
        out = torch.cat([branch4_out, branch3_out, branch2_out], dim=1)  # [B, 128, 1/4H, 1/4W]

        return out

    @staticmethod
    def __make_branch(kernel_size, stride):
        branch = nn.Sequential(
                nn.AvgPool2d(kernel_size, stride),
                Conv2dBn(in_channels=128, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True)  # kernel size maybe 1
            )
        return branch


class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv0 = nn.Sequential(
                Conv2dBn(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, use_relu=True),  # downsample
                Conv2dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True),
                Conv2dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, use_relu=True)
            )

        self.conv1 = StackedBlocks(n_blocks=3, in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv2 = StackedBlocks(n_blocks=16, in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1)  # downsample
        self.conv3 = StackedBlocks(n_blocks=3, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2)  # dilated
        self.conv4 = StackedBlocks(n_blocks=3, in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4)  # dilated

    def forward(self, inputs):
        conv0_out = self.conv0(inputs)
        conv1_out = self.conv1(conv0_out)  # [B, 32, 1/2H, 1/2W]
        conv2_out = self.conv2(conv1_out)  # [B, 64, 1/4H, 1/4W]
        conv3_out = self.conv3(conv2_out)  # [B, 128, 1/4H, 1/4W]
        conv4_out = self.conv4(conv3_out)  # [B, 128, 1/4H, 1/4W]

        return conv2_out, conv4_out


class StackedBlocks(nn.Module):

    def __init__(self, n_blocks, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()

        if stride == 1 and in_channels == out_channels:
            downsample = False
        else:
            downsample = True
        net = [ResidualBlock(in_channels, out_channels, kernel_size, stride, padding, dilation, downsample)]

        for i in range(n_blocks - 1):
            net.append(ResidualBlock(out_channels, out_channels, kernel_size, 1, padding, dilation, downsample=False))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, downsample=False):
        super().__init__()

        self.net = nn.Sequential(
                Conv2dBn(in_channels, out_channels, kernel_size, stride, padding, dilation, use_relu=True),
                Conv2dBn(out_channels, out_channels, kernel_size, 1, padding, dilation, use_relu=False)
            )

        self.downsample = None
        if downsample:
            self.downsample = Conv2dBn(in_channels, out_channels, 1, stride, use_relu=False)

    def forward(self, inputs):
        out = self.net(inputs)
        if self.downsample:
            inputs = self.downsample(inputs)
        out = out + inputs

        return out


class Conv2dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_relu=True):
        super().__init__()

        net = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
               nn.BatchNorm2d(out_channels)]
        if use_relu:
            net.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out

class StackedHourglass(nn.Module):
    '''
    inputs --- [B, 64, 1/4D, 1/4H, 1/4W]
    '''

    def __init__(self, max_disp):
        super().__init__()

        self.conv0 = nn.Sequential(
            Conv3dBn(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True)
        )
        self.conv1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=False)
        )
        self.hourglass1 = Hourglass()
        self.hourglass2 = Hourglass()
        self.hourglass3 = Hourglass()

        self.out1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        self.out2 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )
        self.out3 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True),
            nn.Conv3d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        )

        self.regression = DisparityRegression(max_disp)

    def forward(self, inputs, out_size):

        conv0_out = self.conv0(inputs)     # [B, 32, 1/4D, 1/4H, 1/4W]
        conv1_out = self.conv1(conv0_out)
        conv1_out = conv0_out + conv1_out  # [B, 32, 1/4D, 1/4H, 1/4W]

        hourglass1_out1, hourglass1_out3, hourglass1_out4 = self.hourglass1(conv1_out, scale1=None, scale2=None, scale3=conv1_out)
        hourglass2_out1, hourglass2_out3, hourglass2_out4 = self.hourglass2(hourglass1_out4, scale1=hourglass1_out3, scale2=hourglass1_out1, scale3=conv1_out)
        hourglass3_out1, hourglass3_out3, hourglass3_out4 = self.hourglass3(hourglass2_out4, scale1=hourglass2_out3, scale2=hourglass1_out1, scale3=conv1_out)

        out1 = self.out1(hourglass1_out4)  # [B, 1, 1/4D, 1/4H, 1/4W]
        out2 = self.out2(hourglass2_out4) + out1
        out3 = self.out3(hourglass3_out4) + out2

        cost1 = F.upsample(out1, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]
        cost2 = F.upsample(out2, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]
        cost3 = F.upsample(out3, size=out_size, mode='trilinear').squeeze(dim=1)  # [B, D, H, W]

        prob1 = F.softmax(-cost1, dim=1)  # [B, D, H, W]
        prob2 = F.softmax(-cost2, dim=1)
        prob3 = F.softmax(-cost3, dim=1)

        disp1 = self.regression(prob1)
        disp2 = self.regression(prob2)
        disp3 = self.regression(prob3)

        return disp1, disp2, disp3

class Hourglass(nn.Module):

    def __init__(self):
        super().__init__()

        self.net1 = nn.Sequential(
            Conv3dBn(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=False)
        )
        self.net2 = nn.Sequential(
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, use_relu=True),
            Conv3dBn(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, use_relu=True)
        )
        self.net3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(num_features=64)
            # nn.ReLU(inplace=True)
        )
        self.net4 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm3d(num_features=32)
        )

    def forward(self, inputs, scale1=None, scale2=None, scale3=None):
        net1_out = self.net1(inputs)  # [B, 64, 1/8D, 1/8H, 1/8W]

        if scale1 is not None:
            net1_out = F.relu(net1_out + scale1, inplace=True)
        else:
            net1_out = F.relu(net1_out, inplace=True)

        net2_out = self.net2(net1_out)  # [B, 64, 1/16D, 1/16H, 1/16W]
        net3_out = self.net3(net2_out)  # [B, 64, 1/8D, 1/8H, 1/8W]

        if scale2 is not None:
            net3_out = F.relu(net3_out + scale2, inplace=True)
        else:
            net3_out = F.relu(net3_out + net1_out, inplace=True)

        net4_out = self.net4(net3_out)

        if scale3 is not None:
            net4_out = net4_out + scale3

        return net1_out, net3_out, net4_out

class Conv3dBn(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, use_relu=True):
        super().__init__()

        net = [nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=False),
               nn.BatchNorm3d(out_channels)]
        if use_relu:
            net.append(nn.ReLU(inplace=True))

        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        out = self.net(inputs)
        return out