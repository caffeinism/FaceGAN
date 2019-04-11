# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
# The structure of the code was referenced in cycleGAN github.

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
from config import config

def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type)
    return net


def get_norm_layer(layer_type='instance'):
    if layer_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif layer_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif layer_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % layer_type)
    return norm_layer


def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=False)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(
            nn.LeakyReLU, negative_slope=0.2, inplace=False)
    elif layer_type == 'elu':
        nl_layer = functools.partial(nn.ELU, inplace=False)
    else:
        raise NotImplementedError(
            'nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer


def define_G(input_nc, output_nc, ngf,
             netG='unet_128', norm='batch', nl='relu',
             use_dropout=False, init_type='xavier', gpu_ids=[], upsample='bilinear'):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)

    if netG == 'unet_128':
        net = G_Unet(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample)
    elif netG == 'unet_256':
        net = G_Unet(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, nl_layer=nl_layer,
                               use_dropout=use_dropout, upsample=upsample)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % net)

    return init_net(net, init_type, gpu_ids)


def define_D(input_nc, size, ndf,
             norm='batch', init_type='xavier', num_Ds=1, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(layer_type=norm)
    nl = 'lrelu'  # use leaky relu for D
    nl_layer = get_non_linearity(layer_type=nl)

    net = Discriminator(input_nc, size, norm_layer, nl_layer, ndf)
    return init_net(net, init_type, gpu_ids)
    
class G_Unet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=None, nl_layer=None, use_dropout=False,
                 upsample='basic'):
        super(G_Unet, self).__init__()
        max_nchn = 8
        
        unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn,
                               innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        for i in range(num_downs - 6):
            unet_block = UnetBlock(ngf * max_nchn, ngf * max_nchn, ngf * max_nchn, unet_block,
                                   norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
        unet_block = UnetBlock(ngf * 4, ngf * 4, ngf * max_nchn, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf * 2, ngf * 2, ngf * 4, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(ngf, ngf, ngf * 2, unet_block,
                               norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        unet_block = UnetBlock(input_nc, ngf, ngf, unet_block,
                               outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.model = unet_block

        self.out_conv = nn.Sequential(
            nn.Conv2d(ngf, output_nc, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, face, pose, detach_face=False, detach_pose=False):
        x = self.model(face, pose, detach_face, detach_pose)
        return self.out_conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, norm_layer, nl_layer):
        super(ResidualBlock, self).__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(inplanes, planes, 3, 1, 1, bias=True),
            norm_layer(planes),
            nl_layer(),
            nn.Conv2d(planes, planes, 3, 1, 1, bias=True),
            norm_layer(planes),
        )

        self.nl = nl_layer()

        if inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, 1, 1, 0, bias=False),
                norm_layer(planes),
            )

    def forward(self, x):
        identity = self.shortcut(x) if hasattr(self, 'shortcut') else x

        out = self.residual(x)

        result = self.nl(out + identity)

        return result


def upsampleLayer(inplanes, outplanes, norm_layer, nl_layer, upsample='basic'):
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(
            inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'residual':
        upconv = [ResidualBlock(inplanes, outplanes, norm_layer=norm_layer, nl_layer=nl_layer),
                  nn.Upsample(scale_factor=2, mode='nearest'),
                  ResidualBlock(outplanes, outplanes, norm_layer=norm_layer, nl_layer=nl_layer)]


    else:
        raise NotImplementedError(
            'upsample layer [%s] not implemented' % upsample)
    return upconv


class UnetBlock(nn.Module):
    def __init__(self, input_nc, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False,
                 norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', padding_type='zero'):
        super(UnetBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.input_nc = input_nc
        self.inner_nc = inner_nc
        self.outer_nc = outer_nc

        downconv_face = [nn.Conv2d(input_nc if not outermost else input_nc * 2, inner_nc, kernel_size=4, stride=2, padding=p)]
        downnorm_face = norm_layer(inner_nc)
        downrelu_face = nn.LeakyReLU(0.2, True)
        
        downconv_pose = [nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=p)]
        downnorm_pose = norm_layer(inner_nc)
        downrelu_pose = nn.LeakyReLU(0.2, True)
        
        if outermost:
            down_face = downconv_face + [downrelu_face]
            down_pose = downconv_pose + [downrelu_pose]
            up = upsampleLayer(inner_nc * 3, outer_nc, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)
        elif innermost:
            # flat -> down -> downrelu -> up -> unflat -> upnorm -> uprelu (consider using Dropouts)
            down_face = [nn.Linear(input_nc * 4 * 4, inner_nc), downrelu_face]
            down_pose = [nn.Linear(input_nc * 4 * 4, inner_nc), downrelu_pose]

            up = [nn.Linear(inner_nc * 2, outer_nc * 4 * 4)]

            self.uprelu = nl_layer()
            self.upnorm = norm_layer(outer_nc)
        else:
            down_face = downconv_face + [downnorm_face, downrelu_face]
            down_pose = downconv_pose + [downnorm_pose, downrelu_pose]
            up = upsampleLayer(inner_nc * 3, outer_nc, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample)

        self.down_face = nn.Sequential(*down_face)
        self.down_pose = nn.Sequential(*down_pose)
        self.submodule = submodule
        self.up = nn.Sequential(*up)

    def forward(self, face, pose, detach_face, detach_pose):
        if not self.innermost:
            down_face, down_pose = self.down_face(face), self.down_pose(pose)
            if detach_face: down_face = down_face.detach()
            if detach_pose: down_pose = down_pose.detach()

            sub = self.submodule(down_face, down_pose, detach_face, detach_pose)

            up = self.up(sub)
        else:
            down_face, down_pose = face.view(-1, self.input_nc * 4 * 4), pose.view(-1, self.input_nc * 4 * 4)
            down_face, down_pose = self.down_face(down_face), self.down_pose(down_pose)
            if detach_face: down_face = down_face.detach()
            if detach_pose: down_pose = down_pose.detach()
            
            up = self.up(torch.cat([down_face, down_pose], 1))
            up = up.view(-1, self.outer_nc, 4, 4)
            up = self.upnorm(up)
            up = self.uprelu(up)
            
        if self.outermost:
            return up
        else:
            return torch.cat([up, face, pose], 1)


class Discriminator(nn.Module):
    def __init__(self, input_nc, size, norm_layer, nl_layer, nfilter=64, nfilter_max=512):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        nlayers = int(np.log2(size // s0))
        self.nf0 = min(nf_max, nf * 2**nlayers)

        blocks = [
            ResidualBlock(nf, nf, norm_layer, nl_layer)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResidualBlock(nf0, nf1, norm_layer, nl_layer),
            ]

        self.conv_img = nn.Sequential(nn.Conv2d(input_nc, 1*nf, 3, 1, 1), nl_layer())
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(self.nf0*s0*s0, 1)

    def forward(self, x):
        batch_size = x.size(0)

        out = self.conv_img(x)
        out = self.resnet(out)
        out = out.view(batch_size, self.nf0*self.s0*self.s0)
        out = self.fc(out)

        return out


class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, input):
        loss_tv = torch.mean(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :])) + torch.mean(
                             torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:]))
        return loss_tv


class FeatureL1Loss(nn.Module):
    def __init__(self, weights=None):
        super(FeatureL1Loss, self).__init__()
        self.weights = weights
        self.loss = nn.L1Loss()
    
    def forward(self, input, target):
        assert len(input) == len(target)

        losses = []
        if self.weights == None:
            for a, b in zip(input, target):
                losses.append(self.loss(a, b))
        else:
            assert len(input) == len(self.weights)
            for a, b, weight in zip(input, target, self.weights):
                losses.append(weight * self.loss(a, b))

        return sum(losses) / len(losses)