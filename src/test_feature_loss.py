import torch
from torchvision import models
import define_models_torch as mod
import feature_loss as fl
from torch import nn
import numpy as np
import time

#test if pretrained model and preprocessed iemocap dataset work
iemocap_path = '../dataset/matrices/iemocap_randsplit_spectrum_fast_predictors.npy'
model_path = '../pretraining_vgg/4secs_inv/model'
gpu_ID = 1
device = 'cuda:' + str(gpu_ID)

print ('loading data')
#load input sound from iemicap
iem = np.load(iemocap_path, allow_pickle=True).item()
k = list(iem.keys())
s = k[0]
input = iem[s][0]
input = torch.tensor(input.reshape(1, 1, input.shape[0], input.shape[1])).float().to(device)

#load feature extractor with pretrained weights
#f_extractor, p = mod.vgg16(0,1,['output_classes=1000'])
f_extractor = fl.load_feature_extractor(gpu_ID, '../pretraining_vgg/librispeech/4secs_inv/model', 1000)  #load pretrained weights

#torch.manual_seed(0)
emo_model, p = mod.vgg16(0,1,['output_classes=4'])
emo_model = emo_model.to(device)


x = f_extractor(input)
y = emo_model.features(input)
print ('\nShapes test:')
if x.shape == y.shape:
    print ('    compatible')
else:
    print ('    wrong shape:')
print ('    ' + str(x.shape))
print ('    ' + str(y.shape))

emo_model2, p = mod.vgg16(0,1,['output_classes=4'])
emo_model2 = emo_model.to(device)

#magnitude
mag_f = torch.sum(f_extractor(input))
mag_e = torch.sum(emo_model.features(input))
print ('\nMagnitude test:')
print ('    pretrained: ' + str(mag_f))
print ('    current: ' + str(mag_e))

#range
min_f = torch.min(f_extractor(input))
max_f = torch.max(f_extractor(input))
min_e = torch.min(emo_model.features(input))
max_e = torch.max(emo_model.features(input))
print ('\nRange test:')
print ('    pretrained: min: ' + str(min_f.detach().cpu().numpy()) + '| max: ' + str(max_f.detach().cpu().numpy()))
print ('    current: min: ' + str(min_e.detach().cpu().numpy()) + '| max: ' + str(max_e.detach().cpu().numpy()))

#loss types
print ('\nLoss test:')

start = time.perf_counter()
mse_loss = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'none', distance='mse_sigmoid_invorder')
mse_time = time.perf_counter() - start
print ('    mse: ' + str(mse_loss))
print ('    computation time: ' + str(mse_time))

start = time.perf_counter()
sum_mse = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'sum', distance='mse_sigmoid_invorder')
sum_mse_time = time.perf_counter() - start
print ('    sum_mse: ' + str(sum_mse))
print ('    computation time: ' + str(sum_mse_time))

start = time.perf_counter()
mul_mse = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'mul', distance='mse_sigmoid_invorder')
mul_mse_time = time.perf_counter() - start
print ('    mul_mse: ' + str(mul_mse))
print ('    computation time: ' + str(mul_mse_time))

start = time.perf_counter()
mul_comp001_mse = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'mul_comp001', distance='mse_sigmoid_invorder')
mul_mse_time = time.perf_counter() - start
print ('    mul_comp001_mse: ' + str(mul_mse))
print ('    computation time: ' + str(mul_mse_time))

start = time.perf_counter()
mean_mse = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'mean', distance='mse_sigmoid_invorder')
mean_mse_time = time.perf_counter() - start
print ('    mean_mse: ' + str(mean_mse))
print ('    computation time: ' + str(mean_mse_time))

start = time.perf_counter()
max_mse = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'max', distance='mse_sigmoid_invorder')
max_mse_time = time.perf_counter() - start
print ('    max_mse: ' + str(max_mse))
print ('    computation time: ' + str(max_mse_time))

start = time.perf_counter()
gram_mse = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'gram', distance='mse_sigmoid_invorder')
gram_mse_time = time.perf_counter() - start
print ('    gram_mse: ' + str(gram_mse))
print ('    computation time: ' + str(gram_mse_time))

start = time.perf_counter()
gram_cos = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'gram', distance='cos_squared')
gram_cos_time = time.perf_counter() - start
print ('    gram_cos: ' + str(gram_cos))
print ('    computation time: ' + str(gram_cos_time))
'''
start = time.perf_counter()
pairwise_mse = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'pairwise', distance='mse_sigmoid_invorder')
pairwise_mse_time = time.perf_counter() - start
print ('    pairwise_cos: ' + str(pairwise_mse))
print ('    computation time: ' + str(pairwise_mse_time))

start = time.perf_counter()
pairwise_cos = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'pairwise', distance='cos_squared')
pairwise_cos_time = time.perf_counter() - start
print ('    pairwise_cos: ' + str(pairwise_cos))
print ('    computation time: ' + str(pairwise_cos_time))

start = time.perf_counter()
pairwise_cos_stoc = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'stochastic_pairwise', distance='cos_squared')
pairwise_cos_stoc_time = time.perf_counter() - start
print ('    pairwise_cos: ' + str(pairwise_cos_stoc))
print ('    computation time: ' + str(pairwise_cos_stoc_time))
'''

start = time.perf_counter()
multiple_layers_loss = fl.feature_loss(input, emo_model, f_extractor, aggregation= 'gram', distance='cos_squared', layer=[2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28])
multiple_time = time.perf_counter() - start
print ('    multiple_cos: ' + str(multiple_layers_loss))
print ('    computation time: ' + str(multiple_time))
