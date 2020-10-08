import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn.modules.module import Module
from torch.autograd import Function
from build import GANet

class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()
        #        print(in_channels, out_channels, deconv, is_3d, bn, relu, kwargs)
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True):
        super(Conv2x, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:
            self.conv2 = BasicConv(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        assert (x.size() == rem.size())
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x

class Feature(nn.Module):
    def __init__(self):
        super(Feature, self).__init__()

        self.conv_start = nn.Sequential(
            BasicConv(3, 32, kernel_size=3, padding=1),
            BasicConv(32, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))
        self.conv1a = BasicConv(32, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv(64, 96, kernel_size=3, stride=2, padding=1)
        self.conv4a = BasicConv(96, 128, kernel_size=3, stride=2, padding=1)

        self.deconv4a = Conv2x(128, 96, deconv=True)
        self.deconv3a = Conv2x(96, 64, deconv=True)
        self.deconv2a = Conv2x(64, 48, deconv=True)
        self.deconv1a = Conv2x(48, 32, deconv=True)

        self.conv1b = Conv2x(32, 48)
        self.conv2b = Conv2x(48, 64)
        self.conv3b = Conv2x(64, 96)
        self.conv4b = Conv2x(96, 128)

        self.deconv4b = Conv2x(128, 96, deconv=True)
        self.deconv3b = Conv2x(96, 64, deconv=True)
        self.deconv2b = Conv2x(64, 48, deconv=True)
        self.deconv1b = Conv2x(48, 32, deconv=True)

    def forward(self, x):
        x = self.conv_start(x)
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x
        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)

        return x


class Guidance(nn.Module):
    def __init__(self):
        super(Guidance, self).__init__()

        self.conv0 = BasicConv(64, 16, kernel_size=3, padding=1)
        self.conv1 = nn.Sequential(
            BasicConv(16, 32, kernel_size=5, stride=3, padding=2),
            BasicConv(32, 32, kernel_size=3, padding=1))

        self.conv2 = BasicConv(32, 32, kernel_size=3, padding=1)

        self.conv11 = nn.Sequential(BasicConv(32, 48, kernel_size=3, stride=2, padding=1),
                                    BasicConv(48, 48, kernel_size=3, padding=1))
        self.conv12 = BasicConv(48, 48, kernel_size=3, padding=1)

        self.weight_sg1 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg2 = nn.Conv2d(32, 640, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_sg11 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)
        self.weight_sg12 = nn.Conv2d(48, 960, (3, 3), (1, 1), (1, 1), bias=False)

        self.weight_lg1 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))
        self.weight_lg2 = nn.Sequential(BasicConv(16, 16, kernel_size=3, padding=1),
                                        nn.Conv2d(16, 75, (3, 3), (1, 1), (1, 1), bias=False))

    def forward(self, x):
        x = self.conv0(x)
        rem = x
        x = self.conv1(x)
        sg1 = self.weight_sg1(x)
        x = self.conv2(x)
        sg2 = self.weight_sg2(x)

        x = self.conv11(x)
        sg11 = self.weight_sg11(x)
        x = self.conv12(x)
        sg12 = self.weight_sg12(x)

        lg1 = self.weight_lg1(rem)
        lg2 = self.weight_lg2(rem)

        return dict([
            ('sg1', sg1),
            ('sg2', sg2),
            ('sg11', sg11),
            ('sg12', sg12),
            ('lg1', lg1),
            ('lg2', lg2)])

class Disp(nn.Module):

    def __init__(self, maxdisp=192):
        super(Disp, self).__init__()
        self.maxdisp = maxdisp
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)
#        self.conv32x1 = BasicConv(32, 1, kernel_size=3)
        self.conv32x1 = nn.Conv3d(32, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

    def forward(self, x):
        x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        x = self.softmax(x)

        return self.disparity(x)

class DisparityRegression(Module):
    def __init__(self, maxdisp):
       super(DisparityRegression, self).__init__()
       self.maxdisp = maxdisp + 1
#        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1,self.maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        assert(x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            disp = Variable(torch.Tensor(np.reshape(np.array(range(self.maxdisp)),[1, self.maxdisp, 1, 1])).cuda(), requires_grad=False)
            disp = disp.repeat(x.size()[0], 1, x.size()[2], x.size()[3])
            out = torch.sum(x * disp, 1)
        return out

class DispAgg(nn.Module):

    def __init__(self, maxdisp=192):
        super(DispAgg, self).__init__()
        self.maxdisp = maxdisp
        self.LGA3 = LGA3(radius=2)
        self.LGA2 = LGA2(radius=2)
        self.LGA = LGA(radius=2)
        self.softmax = nn.Softmin(dim=1)
        self.disparity = DisparityRegression(maxdisp=self.maxdisp)
#        self.conv32x1 = BasicConv(32, 1, kernel_size=3)
        self.conv32x1=nn.Conv3d(32, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1), bias=False)

    def lga(self, x, g):
        g = F.normalize(g, p=1, dim=1)
        x = self.LGA2(x, g)
        return x

    def forward(self, x, lg1, lg2):
        x = F.interpolate(self.conv32x1(x), [self.maxdisp+1, x.size()[3]*3, x.size()[4]*3], mode='trilinear', align_corners=False)
        x = torch.squeeze(x, 1)
        assert(lg1.size() == lg2.size())
        x = self.lga(x, lg1)
        x = self.softmax(x)
        x = self.lga(x, lg2)
        x = F.normalize(x, p=1, dim=1)
        return self.disparity(x)

class LGA3(Module):
    def __init__(self, radius=2):
        super(LGA3, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga3Function.apply(input1, input2, self.radius)
        return result
class LGA2(Module):
    def __init__(self, radius=2):
        super(LGA2, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = Lga2Function.apply(input1, input2, self.radius)
        return result
class LGA(Module):
    def __init__(self, radius=2):
        super(LGA, self).__init__()
        self.radius = radius

    def forward(self, input1, input2):
        result = LgaFunction.apply(input1, input2, self.radius)
        return result

class Lga3Function(Function):
    @staticmethod
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            temp_out1 = input.new().resize_(num, channels, height, width).zero_()
            temp_out2 = input.new().resize_(num, channels, height, width).zero_()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, temp_out1, radius)
            GANet.lga_cuda_forward(temp_out1, filters, temp_out2, radius)
            GANet.lga_cuda_forward(temp_out2, filters, output, radius)
            output = output.contiguous()
        ctx.save_for_backward(input, filters, temp_out1, temp_out2)
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, filters, temp_out1, temp_out2 = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = input.size()
            _, fsize, _, _ = filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(temp_out2, filters, gradOutput, temp_out2, gradFilters, ctx.radius)
            GANet.lga_cuda_backward(temp_out1, filters, temp_out2, temp_out1, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lga_cuda_backward(input, filters, temp_out1, temp_out2, gradFilters, ctx.radius)
#            temp_out[...] = gradOutput[...]
            temp_out2 = temp_out2.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out2, gradFilters, None
class Lga2Function(Function):
    @staticmethod
    def forward(ctx, input, filters, radius=1):
        ctx.radius = radius
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            temp_out = input.new().resize_(num, channels, height, width).zero_()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, temp_out, radius)
            GANet.lga_cuda_forward(temp_out, filters, output, radius)
            output = output.contiguous()
        ctx.save_for_backward(input, filters, temp_out)
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, filters, temp_out = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = input.size()
            _, fsize, _, _ = filters.size()
#            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(temp_out, filters, gradOutput, temp_out, gradFilters, ctx.radius)
#            temp_out[...] = 0
            GANet.lga_cuda_backward(input, filters, temp_out, gradOutput, gradFilters, ctx.radius)
            temp_out[...] = gradOutput[...]
            temp_out = temp_out.contiguous()
            gradFilters = gradFilters.contiguous()
        return temp_out, gradFilters, None
class LgaFunction(Function):
    @staticmethod
    def forward(ctx, input, filters):
        ctx.radius = radius
        assert(input.is_contiguous() == True and filters.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, height, width = input.size()
            output = input.new().resize_(num, channels, height, width).zero_()
            GANet.lga_cuda_forward(input, filters, output, radius)
            output = output.contiguous()
        ctx.save_for_backward(input, filters)
        return output
    @staticmethod
    def backward(ctx, gradOutput):
        input, filters = ctx.saved_tensors
        assert(gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, height, width = input.size()
            _, fsize, _, _ = filters.size()
            gradInput = gradOutput.new().resize_(num, channels, height, width).zero_()
            gradFilters = gradOutput.new().resize_(num, fsize, height, width).zero_()
            GANet.lga_cuda_backward(input, filters, gradOutput, gradInput, gradFilters, ctx.radius)
            gradInput = gradInput.contiguous()
            gradFilters = gradFilters.contiguous()
        return gradInput, gradFilters, None

class SGABlock(nn.Module):
    def __init__(self, channels=32, refine=False):
        super(SGABlock, self).__init__()
        self.refine = refine
        if self.refine:
            self.bn_relu = nn.Sequential(nn.BatchNorm3d(channels),
                                         nn.ReLU(inplace=True))
            self.conv_refine = BasicConv(channels, channels, is_3d=True, kernel_size=3, padding=1, relu=False)
#            self.conv_refine1 = BasicConv(8, 8, is_3d=True, kernel_size=1, padding=1)
        else:
            self.bn = nn.BatchNorm3d(channels)
        self.SGA=SGA()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, g):
        rem = x
        k1, k2, k3, k4 = torch.split(g, (x.size()[1]*5, x.size()[1]*5, x.size()[1]*5, x.size()[1]*5), 1)
        k1 = F.normalize(k1.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k2 = F.normalize(k2.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k3 = F.normalize(k3.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        k4 = F.normalize(k4.view(x.size()[0], x.size()[1], 5, x.size()[3], x.size()[4]), p=1, dim=2)
        x = self.SGA(x, k1, k2, k3, k4)
        if self.refine:
            x = self.bn_relu(x)
            x = self.conv_refine(x)
        else:
            x = self.bn(x)
        assert(x.size() == rem.size())
        x += rem
        return self.relu(x)
#        return self.bn_relu(x)

class SGA(Module):
    def __init__(self):
        super(SGA, self).__init__()

    def forward(self, input, g0, g1, g2, g3):
        result = SgaFunction.apply(input, g0, g1, g2, g3)
        return result


class SgaFunction(Function):
    @staticmethod
    def forward(ctx, input, g0, g1, g2, g3):
        assert (
                    input.is_contiguous() == True and g0.is_contiguous() == True and g1.is_contiguous() == True and g2.is_contiguous() == True and g3.is_contiguous() == True)
        with torch.cuda.device_of(input):
            num, channels, depth, height, width = input.size()
            output = input.new().resize_(num, channels, depth, height, width).zero_()
            temp_out = input.new().resize_(num, channels, depth, height, width).zero_()
            mask = input.new().resize_(num, channels, depth, height, width).zero_()
            GANet.sga_cuda_forward(input, g0, g1, g2, g3, temp_out, output, mask)
            #           GANet.sga_cuda_forward(input, filters, output, radius)

            output = output.contiguous()
        ctx.save_for_backward(input, g0, g1, g2, g3, temp_out, mask)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        input, g0, g1, g2, g3, temp_out, mask = ctx.saved_tensors
        #        print temp_out.size()
        #        print mask.size()
        assert (gradOutput.is_contiguous() == True)
        with torch.cuda.device_of(gradOutput):
            num, channels, depth, height, width = input.size()
            #            _, _, fsize, _, _ = g0.size()
            #            print fsize
            gradInput = gradOutput.new().resize_(num, channels, depth, height, width).zero_()
            grad0 = gradOutput.new().resize_(g0.size()).zero_()
            grad1 = gradOutput.new().resize_(g1.size()).zero_()
            grad2 = gradOutput.new().resize_(g2.size()).zero_()
            grad3 = gradOutput.new().resize_(g3.size()).zero_()
            temp_grad = gradOutput.new().resize_(num, channels, depth, height, width).zero_()
            max_idx = gradOutput.new().resize_(num, channels, height, width).zero_()

            GANet.sga_cuda_backward(input, g0, g1, g2, g3, temp_out, mask, max_idx, gradOutput, temp_grad, gradInput,
                                    grad0, grad1, grad2, grad3)
            #            GANet.lga_cuda_backward(input, filters, gradOutput, gradInput, gradFilters, radius)
            gradInput = gradInput.contiguous()
            grad0 = grad0.contiguous()
            grad1 = grad1.contiguous()
            grad2 = grad2.contiguous()
            grad3 = grad3.contiguous()
        return gradInput, grad0, grad1, grad2, grad3


class CostAggregation(nn.Module):
    def __init__(self, maxdisp=192):
        super(CostAggregation, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = BasicConv(64, 32, is_3d=True, kernel_size=3, padding=1, relu=False)

        self.conv1a = BasicConv(32, 48, is_3d=True, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv(48, 64, is_3d=True, kernel_size=3, stride=2, padding=1)
        #        self.conv3a = BasicConv(64, 96, is_3d=True, kernel_size=3, stride=2, padding=1)

        self.deconv1a = Conv2x(48, 32, deconv=True, is_3d=True, relu=False)
        self.deconv2a = Conv2x(64, 48, deconv=True, is_3d=True)
        #        self.deconv0a = Conv2x(8, 8, deconv=True, is_3d=True)

        self.sga1 = SGABlock(refine=True)
        self.sga2 = SGABlock(refine=True)

        self.sga11 = SGABlock(channels=48, refine=True)
        self.sga12 = SGABlock(channels=48, refine=True)

        self.disp0 = Disp(self.maxdisp)
        self.disp1 = DispAgg(self.maxdisp)

    def forward(self, x, g):

        x = self.conv_start(x)
        x = self.sga1(x, g['sg1'])
        rem0 = x

        if self.training:
            disp0 = self.disp0(x)

        x = self.conv1a(x)
        x = self.sga11(x, g['sg11'])
        rem1 = x
        x = self.conv2a(x)
        x = self.deconv2a(x, rem1)
        x = self.sga12(x, g['sg12'])
        x = self.deconv1a(x, rem0)
        x = self.sga2(x, g['sg2'])
        disp1 = self.disp1(x, g['lg1'], g['lg2'])

        if self.training:
            return disp0, disp1
        else:
            return disp1



class GetCostVolume(Module):
    def __init__(self, maxdisp):
        super(GetCostVolume, self).__init__()
        self.maxdisp = maxdisp + 1

    def forward(self, x, y):
        assert (x.is_contiguous() == True)
        with torch.cuda.device_of(x):
            num, channels, height, width = x.size()
            cost = x.new().resize_(num, channels * 2, self.maxdisp, height, width).zero_()
            #            cost = Variable(torch.FloatTensor(x.size()[0], x.size()[1]*2, self.maxdisp,  x.size()[2],  x.size()[3]).zero_(), volatile= not self.training).cuda()
            for i in range(self.maxdisp):
                if i > 0:
                    cost[:, :x.size()[1], i, :, i:] = x[:, :, :, i:]
                    cost[:, x.size()[1]:, i, :, i:] = y[:, :, :, :-i]
                else:
                    cost[:, :x.size()[1], i, :, :] = x
                    cost[:, x.size()[1]:, i, :, :] = y

            cost = cost.contiguous()
        return cost


class GA_Net(nn.Module):
    def __init__(self, maxdisp=192):
        super(GA_Net, self).__init__()
        self.maxdisp = maxdisp
        self.conv_start = nn.Sequential(BasicConv(3, 16, kernel_size=3, padding=1),
                                        BasicConv(16, 32, kernel_size=3, padding=1))

        self.conv_x = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_y = BasicConv(32, 32, kernel_size=3, padding=1)
        self.conv_refine = nn.Conv2d(32, 32, (3, 3), (1, 1), (1, 1), bias=False)
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(32),
                                     nn.ReLU(inplace=True))
        self.feature = Feature()
        self.guidance = Guidance()
        self.cost_agg = CostAggregation(self.maxdisp)
        self.cv = GetCostVolume(int(self.maxdisp / 3))

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, y):
        g = self.conv_start(x)
        x = self.feature(x)

        rem = x
        x = self.conv_x(x)

        y = self.feature(y)
        y = self.conv_y(y)

        x = self.cv(x, y)

        x1 = self.conv_refine(rem)
        x1 = F.interpolate(x1, [x1.size()[2] * 3, x1.size()[3] * 3], mode='bilinear', align_corners=False)
        x1 = self.bn_relu(x1)
        g = torch.cat((g, x1), 1)
        g = self.guidance(g)

        return self.cost_agg(x, g)















