import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as utils
import define_models_torch as mod
import numpy as np
from torchvision import models
import configparser
import loadconfig

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
torch.manual_seed(3)
at_model_path = cfg.get('training_defaults', 'at_model_path')
num_cnn_layers = cfg.getint('pretraining', 'num_cnn_layers')
pretraining_classes_load = cfg.getint('training_defaults', 'pretraining_classes_load')

cos = nn.CosineSimilarity(dim=-1)

#load model

parameters = ['output_classes = ' + str(pretraining_classes_load)]
model, p = mod.vgg16(0,0, parameters)

epsylon = 1e-10

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL
    G = torch.mm(features, features.t())  # compute the gram product
    return G.div(a * b * c * d)

def pairwise_aggregation(a, b):
    a = a.view(a.shape[0], a.shape[1], a.shape[2]*a.shape[3])
    b = b.view(b.shape[0], b.shape[1], b.shape[2]*b.shape[3])
    ch = a.shape[1]

    a = a.repeat(1, 1, ch)
    a = a.view(a.shape[0], 1, a.shape[1]*a.shape[2])

    b = b.view(b.shape[0], 1, b.shape[1]*b.shape[2])
    b = b.repeat(1, 1, ch)

    return a, b

def stochastic_pairwise_aggregation(a, b):
    b_dim, c_dim, x_dim, y_dim = a.shape

    if b_dim > 1:
        q = int(c_dim / b_dim)  #chans/batch so that 1 epoch contains all channels
        begin = np.random.randint(c_dim - q)
        end = begin + q
    else:
        begin = np.random.randint(c_dim)
        end = begin + 1

    a = a[:,begin:end,:,:]
    b = b[:,begin:end,:,:]

    a = a.view(a.shape[0], a.shape[1], a.shape[2]*a.shape[3])
    b = b.view(b.shape[0], b.shape[1], b.shape[2]*b.shape[3])
    ch = a.shape[1]

    a = a.repeat(1, 1, ch)
    a = a.view(a.shape[0], 1, a.shape[1]*a.shape[2])

    b = b.view(b.shape[0], 1, b.shape[1]*b.shape[2])
    b = b.repeat(1, 1, ch)
    #print ('culo', torch.sum(a), torch.sum(b))

    return a, b

def pairwise_cos_squared(a, b):
    b_dim, c_dim, x_dim, y_dim = a.shape
    div = b_dim * c_dim * c_dim
    loss = torch.tensor(0).float()
    for batch in range(b_dim):
        batch_a = a[batch]
        batch_b = b[batch]
        for channel in range(c_dim):
            channel_a = batch_a[channel].flatten()
            for paired in range(c_dim):
                channel_b = batch_b[paired].flatten()
                temp_loss = torch.abs(cos(channel_a, channel_b))
                loss += temp_loss
    loss = loss / div

    return loss

def load_feature_extractor(gpu_id, model=model):
    device = torch.device('cuda:' + str(gpu_id))
    model.load_state_dict(torch.load(at_model_path,
                        map_location=lambda storage, location: storage),
                        strict=False)
    model = model.eval().features.to(device)
    return model

def feature_loss(input, current_model, pretrained_model, beta=1., layer=28,
                aggregation='none', distance='mse_sigmoid'):
    curr_feat = current_model.features[:layer](input)
    pre_feat = pretrained_model[:layer](input)

    if aggregation == 'none':
        pass

    elif aggregation == 'mean':
        curr_feat = curr_feat.mean(1).view(curr_feat.shape[0], 1,
                    curr_feat.shape[2], curr_feat.shape[3])
        pre_feat = pre_feat.mean(1).view(pre_feat.shape[0], 1,
                    pre_feat.shape[2], pre_feat.shape[3])

    elif aggregation == 'sum':
        curr_feat = curr_feat.sum(1).view(curr_feat.shape[0], 1,
                    curr_feat.shape[2], curr_feat.shape[3])
        pre_feat = pre_feat.sum(1).view(pre_feat.shape[0], 1,
                    pre_feat.shape[2], pre_feat.shape[3])

    elif aggregation == 'mul':
        curr_feat = curr_feat + epsylon
        pre_feat = pre_feat + epsylon
        curr_feat = curr_feat.prod(1).view(curr_feat.shape[0], 1,
                    curr_feat.shape[2], curr_feat.shape[3])
        pre_feat = pre_feat.prod(1).view(pre_feat.shape[0], 1,
                    pre_feat.shape[2], pre_feat.shape[3])

    elif aggregation == 'mul_comp001':
        curr_feat = curr_feat + epsylon
        pre_feat = pre_feat + epsylon
        #print ('QUIIIIII', torch.sum(curr_feat), torch.sum(pre_feat))
        curr_feat = curr_feat ** 0.001
        pre_feat = pre_feat ** 0.001
        #print ('QUIIIIII2', torch.sum(curr_feat), torch.sum(pre_feat))
        curr_feat = curr_feat.prod(1).view(curr_feat.shape[0], 1,
                    curr_feat.shape[2], curr_feat.shape[3])
        pre_feat = pre_feat.prod(1).view(pre_feat.shape[0], 1,
                    pre_feat.shape[2], pre_feat.shape[3])
        #print ('QUIIIIII3', torch.sum(curr_feat), torch.sum(pre_feat))


    elif aggregation == 'mul_comp005':
        curr_feat = curr_feat + epsylon
        pre_feat = pre_feat + epsylon
        curr_feat = curr_feat ** 0.005
        pre_feat = pre_feat ** 0.005
        curr_feat = curr_feat.prod(1).view(curr_feat.shape[0], 1,
                    curr_feat.shape[2], curr_feat.shape[3])
        pre_feat = pre_feat.prod(1).view(pre_feat.shape[0], 1,
                    pre_feat.shape[2], pre_feat.shape[3])

    elif aggregation == 'mul_comp01':
        curr_feat = curr_feat + epsylon
        pre_feat = pre_feat + epsylon
        curr_feat = curr_feat ** 0.01
        pre_feat = pre_feat ** 0.01
        curr_feat = curr_feat.prod(1).view(curr_feat.shape[0], 1,
                    curr_feat.shape[2], curr_feat.shape[3])
        pre_feat = pre_feat.prod(1).view(pre_feat.shape[0], 1,
                    pre_feat.shape[2], pre_feat.shape[3])
    elif aggregation == 'mul_comp05':
        curr_feat = curr_feat + epsylon
        pre_feat = pre_feat + epsylon
        curr_feat = curr_feat ** 0.05
        pre_feat = pre_feat ** 0.05
        curr_feat = curr_feat.prod(1).view(curr_feat.shape[0], 1,
                    curr_feat.shape[2], curr_feat.shape[3])
        pre_feat = pre_feat.prod(1).view(pre_feat.shape[0], 1,
                    pre_feat.shape[2], pre_feat.shape[3])
    elif aggregation == 'mul_comp1':
        curr_feat = curr_feat + epsylon
        pre_feat = pre_feat + epsylon
        curr_feat = curr_feat ** 0.1
        pre_feat = pre_feat ** 0.1
        curr_feat = curr_feat.prod(1).view(curr_feat.shape[0], 1,
                    curr_feat.shape[2], curr_feat.shape[3])
        pre_feat = pre_feat.prod(1).view(pre_feat.shape[0], 1,
                    pre_feat.shape[2], pre_feat.shape[3])

    elif aggregation == 'max':
        curr_feat = F.max_pool3d(curr_feat, kernel_size=[curr_feat.shape[1],1,1])
        pre_feat = F.max_pool3d(pre_feat, kernel_size=[pre_feat.shape[1],1,1])

    elif aggregation == 'gram':
        curr_feat = gram_matrix(curr_feat) / 0.001
        pre_feat = gram_matrix(pre_feat) / 0.001

    elif aggregation == 'pairwise':
        curr_feat, pre_feat = pairwise_aggregation(curr_feat, pre_feat)

    elif aggregation == 'stochastic_pairwise':
        curr_feat, pre_feat = stochastic_pairwise_aggregation(curr_feat, pre_feat)

    else:
        raise NameError('wrong aggregation type selected')

    print (' culo', torch.sum(curr_feat).item(), torch.sum(pre_feat).item())


    if distance == 'mse_sigmoid':
        loss = F.mse_loss(curr_feat, pre_feat)
        loss = F.sigmoid(loss)
        loss = loss * -1
        loss = loss * beta

    if distance == 'mse_sigmoid_invorder':
        loss = F.mse_loss(curr_feat, pre_feat)
        loss = loss * beta
        loss = loss * -1
        loss = F.sigmoid(loss)

    elif distance == 'cos_squared':
        if len(curr_feat.shape) > 3:
            curr_feat = curr_feat.view(curr_feat.shape[0], curr_feat.shape[1], curr_feat.shape[2]*curr_feat.shape[3])
            pre_feat = pre_feat.view(pre_feat.shape[0], pre_feat.shape[1], pre_feat.shape[2]*pre_feat.shape[3])
        loss = cos(curr_feat, pre_feat)
        loss = torch.abs(loss)
        loss = torch.mean(loss)
        loss = loss ** 2
        loss = loss * beta

    elif distance == 'pairwise_cos_squared':
        loss = pairwise_cos_squared(curr_feat, pre_feat)
        loss = loss ** 2
        loss = loss * beta

    else:
        raise NameError('wrong distance type selected')



    return loss

'''
pretrained_model = load_feature_extractor(0)
current_model = models.vgg16().to('cuda:0')
#print (repr(current_model))
input = torch.rand(2,3,129,170).to('cuda:0')
feature_loss(input, current_model, pretrained_model, 0.5)
'''


#prova con iemocap
