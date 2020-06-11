from __future__ import print_function
import numpy as np
import configparser
import matplotlib.pyplot as plt
from torchvision import models
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from multiscale_convlayer2 import MultiscaleConv2d
import sys, os
import configparser
import loadconfig


config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
PRETRAINED_MODELS_FOLDER = cfg.get('training_defaults', 'pretrained_models_folder')

def parse_parameters(defaults, parameters):
    for param in parameters:
        param = param.split('=')
        item = param[0].replace(' ', '')
        value = eval(param[1].replace(' ', ''))
        defaults[item] = value
    return defaults


#DEFINE HERE YOUR MODELS!!
def CNN_1conv(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'channels':1,
    'kernel_size_1': [10,5],
    'output_type': 'pooled_map',
    'stretch_penality_lambda': 0.,
    'stretch_factors': [],
    'hidden_size': 200,
    'fc_insize':100,
    'output_classes':8,
    'dropout': False,
    'drop_prob': 0.4
    }

    p = parse_parameters(p, user_parameters)

    #always return model AND p!!!
    class CNN_1conv_class(nn.Module):

        def __init__(self):
            super(CNN_1conv_class, self).__init__()
            self.layer_type = p['layer_type']
            self.inner_state = True
            if self.layer_type == 'conv':
                self.conv1 = nn.Conv2d(1, p['channels'], kernel_size=p['kernel_size_1'])
            if self.layer_type == 'multi':
                self.multiscale1 = MultiscaleConv2d(1, p['channels'], kernel_size=p['kernel_size_1'], scale_factors=p['stretch_factors'],
                                            output_type=p['output_type'], stretch_penality_lambda=p['stretch_penality_lambda'])
            self.hidden = nn.Linear(p['fc_insize'], p['hidden_size'])
            self.out = nn.Linear(p['hidden_size'], p['output_classes'])
            self.dropout = p['dropout']
            self.drop_prob = p['drop_prob']

        def forward(self, X):
            training_state = self.training
            if self.layer_type == 'conv':
                X = F.relu(self.conv1(X))
            if self.layer_type == 'multi':
                X = F.relu(self.multiscale1(X, training_state))
            X = X.reshape(X.size(0), -1)
            if self.dropout:
                X = F.dropout2d(X, self.drop_prob)
            X = F.relu(self.hidden(X))
            X = self.out(X)

            return X


    out = CNN_1conv_class()

    return out, p


def CNN_2conv(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'channels':10,
    'kernel_size_1': [10,5],
    'output_type': 'pooled_map',
    'stretch_penality_lambda': 0.,
    'stretch_factors': [],
    'hidden_size': 200,
    'fc_insize':100,
    'output_classes':8,
    'dropout': False,
    'drop_prob': 0.4,
    'pool_size': [2,2]
    }

    p = parse_parameters(p, user_parameters)

    #always return model AND p!!!
    class CNN_2conv_class(nn.Module):
        def __init__(self):
            super(CNN_2conv_class, self).__init__()
            self.layer_type = p['layer_type']
            self.inner_state = True
            if p['layer_type'] == 'conv':
                self.conv1 = nn.Conv2d(1, p['channels'], kernel_size=p['kernel_size_1'])
                self.conv2 = nn.Conv2d(p['channels'], p['channels'], kernel_size=p['kernel_size_1'])
            if p['layer_type'] == 'multi':
                self.multiscale1 = MultiscaleConv2d(1, p['channels'], kernel_size=p['kernel_size_1'], scale_factors=p['stretch_factors'],
                                                output_type=p['output_type'], stretch_penality_lambda= p['stretch_penality_lambda'])
                self.multiscale2 = MultiscaleConv2d(p['channels'], p['channels'], kernel_size=p['kernel_size_1'], scale_factors=p['stretch_factors'],
                                                output_type=p['output_type'], stretch_penality_lambda= p['stretch_penality_lambda'])
            self.pool = nn.MaxPool2d(p['pool_size'][0], p['pool_size'][1])
            self.hidden = nn.Linear(p['fc_insize'], p['hidden_size'])
            self.out = nn.Linear(p['hidden_size'], p['output_classes'])
            self.dropout = p['dropout']
            self.drop_prob = p['drop_prob']

        def forward(self, X):
            training_state = self.training
            if self.layer_type == 'conv':
                X = F.relu(self.conv1(X))
                X = self.pool(X)
                X = F.relu(self.conv2(X))
            if self.layer_type == 'multi':
                X = F.relu(self.multiscale1(X, training_state))
                X = self.pool(X)
                X = F.relu(self.multiscale2(X, training_state))
            X = X.reshape(X.size(0), -1)
            if self.dropout:
                X = F.dropout2d(X, self.drop_prob)
            X = F.relu(self.hidden(X))
            X = self.out(X)

            return X


    out = CNN_2conv_class()

    return out, p

def AlexNet(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'output_type': 'pooled_map',
    'stretch_penality_lambda': 0.,
    'stretch_factors': [],
    'fc_insize':100,
    'output_classes':8,
    'multiplier': 1
    }

    p = parse_parameters(p, user_parameters)

    #always return model AND p!!!
    #always return model AND p!!!
    class AlexNet_class(nn.Module):
        def __init__(self, num_classes=p['output_classes']):
            super(AlexNet_class, self).__init__()
            if p['layer_type'] == 'conv':
                self.features = nn.Sequential(
                    nn.Conv2d(1, int(64*p['multiplier']), kernel_size=11, stride=4, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(int(64*p['multiplier']), int(192*p['multiplier']), kernel_size=5, padding=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(int(192*p['multiplier']), int(384*p['multiplier']), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(int(384*p['multiplier']), int(256*p['multiplier']), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(int(256*p['multiplier']), int(256*p['multiplier']), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )
            if p['layer_type'] == 'multi':
                self.features = nn.Sequential(
                    MultiscaleConv2d(1, int(64*p['multiplier']), kernel_size=[11,11], scale_factors=p['stretch_factors'], padding=2,
                                    output_type=p['output_type'], stretch_penality_lambda= p['stretch_penality_lambda']),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                                        MultiscaleConv2d(int(64*p['multiplier']),int(192*p['multiplier']), kernel_size=[5,5], scale_factors=p['stretch_factors'], padding=2,
                                                        output_type=p['output_type'], stretch_penality_lambda= p['stretch_penality_lambda']),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    nn.Conv2d(int(192*p['multiplier']), int(384*p['multiplier']), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(int(384*p['multiplier']), int(256*p['multiplier']), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(int(256*p['multiplier']), int(256*p['multiplier']), kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2),
                )

            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(int(256 * 6 * 6*p['multiplier']), 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    out = AlexNet_class()

    return out, p


def ResNet18(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'channels':1,
    'kernel_size_1': [10,5],
    'output_type': 'pooled_map',
    'stretch_penality_lambda': 0.,
    'stretch_factors': [],
    'hidden_size': 200,
    'fc_insize':100,
    'output_classes':8,
    'dropout': False,
    'drop_prob': 0.4
    }

    p = parse_parameters(p, user_parameters)

    #always return model AND p!!!


    def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                         padding=dilation, groups=groups, bias=False, dilation=dilation)


    def conv1x1(in_planes, out_planes, stride=1):
        """1x1 convolution"""
        return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


    class BasicBlock(nn.Module):
        expansion = 1
        __constants__ = ['downsample']

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                     base_width=64, dilation=1, norm_layer=None):
            super(BasicBlock, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv3x3(inplanes, planes, stride)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = conv3x3(planes, planes)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


    class Bottleneck(nn.Module):
        expansion = 4
        __constants__ = ['downsample']

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                     base_width=64, dilation=1, norm_layer=None):
            super(Bottleneck, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = conv1x1(inplanes, width)
            self.bn1 = norm_layer(width)
            self.conv2 = conv3x3(width, width, stride, groups, dilation)
            self.bn2 = norm_layer(width)
            self.conv3 = conv1x1(width, planes * self.expansion)
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out

    """## Resnet with Multiscale layers"""

    class BasicBlock_MS(nn.Module):
        expansion = 1
        __constants__ = ['downsample']

        def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                     base_width=64, dilation=1, norm_layer=None, scale_factors=[(0.7,1.),(1.428,1.0)],
                     output_type='pooled_map', stretch_penality_lambda=0., padding=1):
            super(BasicBlock_MS, self).__init__()
            self.groups = groups
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            if groups != 1 or base_width != 64:
                raise ValueError('BasicBlock_MS only supports groups=1 and base_width=64')
            if dilation > 1:
                raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
            # Both self.conv1 and self.downsample layers downsample the input when stride != 1
            self.conv1 = MultiscaleConv2d(inplanes, planes, kernel_size=(3,3), stride=stride, padding=padding, scale_factors=scale_factors)
            self.bn1 = norm_layer(planes)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = MultiscaleConv2d(planes, planes, kernel_size=(3,3), stride=1, padding=padding, scale_factors=scale_factors)
            self.bn2 = norm_layer(planes)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


    class Bottleneck_MS(nn.Module):
        expansion = 4
        __constants__ = ['downsample']

        def __init__(self, inplanes, planes, stride=1,
                     downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None,
                     scale_factors=[(0.7, 1.), (1.428, 1.0)], output_type='pooled_map', stretch_penality_lambda=0.):
            super(Bottleneck_MS, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            width = int(planes * (base_width / 64.)) * groups
            # Both self.conv2 and self.downsample layers downsample the input when stride != 1
            self.conv1 = MultiscaleConv2d(inplanes, width, kernel_size(1,1))
            self.bn1 = norm_layer(width)
            self.conv2 = MultiscaleConv2d(width, width, kernel_size=(3,3), stride=stride, padding=1, scale_factors=scale_factors)
            self.bn2 = norm_layer(width)
            self.conv3 = MultiscaleConv2d(width, planes * self.expansion, kernel_size(1,1))
            self.bn3 = norm_layer(planes * self.expansion)
            self.relu = nn.ReLU(inplace=True)
            self.downsample = downsample
            self.stride = stride

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity
            out = self.relu(out)

            return out


    class ResNet(nn.Module):

        def __init__(self, conv_type=p['layer_type'], layers = [2, 2, 2, 2], num_classes=p['output_classes'], zero_init_residual=False,\
                     groups=1, width_per_group=64, replace_stride_with_dilation=None,\
                     norm_layer=None, scale_factors=p['stretch_factors'], output_type=p['output_type'], stretch_penality_lambda=p['stretch_penality_lambda']):
            super(ResNet, self).__init__()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self._norm_layer = norm_layer
            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            self.groups = groups
            self.base_width = width_per_group
            if conv_type == 'multi':
              self.conv1 = MultiscaleConv2d(3, self.inplanes, kernel_size=(7,7), stride=2, padding=3, scale_factors=scale_factors)
              block = BasicBlock_MS
            elif conv_type == 'conv':
              self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
              block = BasicBlock
            else:
              raise InputError("For the parameter 'conv_type' please enter either 'multi' or 'conv'")
            self.bn1 = norm_layer(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

            # Zero-initialize the last BN in each residual branch,
            # so that the residual branch starts with zeros, and each residual block behaves like an identity.
            # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
            if zero_init_residual:
                for m in self.modules():
                  if conv_type == 'multi':
                    if isinstance(m, Bottleneck_MS):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock_MS):
                        nn.init.constant_(m.bn2.weight, 0)
                  elif conv_type == 'conv':
                    if isinstance(m, Bottleneck):
                        nn.init.constant_(m.bn3.weight, 0)
                    elif isinstance(m, BasicBlock):
                        nn.init.constant_(m.bn2.weight, 0)

        def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
            norm_layer = self._norm_layer
            downsample = None
            previous_dilation = self.dilation
            if dilate:
                self.dilation *= stride
                stride = 1
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )

            layers = []
            layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                self.base_width, previous_dilation, norm_layer))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = F.interpolate(x, size=[224,224], mode='bilinear')
            x = torch.cat((x,x,x), axis=1)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

            return x


    model_class = ResNet()

    return model_class, p

def vgg16(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'output_classes':1000,

    }
    model = models.vgg16()

    p = parse_parameters(p, user_parameters)

    #1 channel input instead of 3
    model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3),
                                stride=(1, 1), padding=(1, 1))

    #change num output classes
    model.classifier[6] =nn.Linear(in_features=4096,
                                out_features=p['output_classes'], bias=True)


    out = model.cpu()

    return out, p
