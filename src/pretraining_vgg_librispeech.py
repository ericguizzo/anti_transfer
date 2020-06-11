import sys, os
import torch
import torch.nn as nn
from torch import optim
import torch.utils.data as utils
import numpy as np
from torchvision import models
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import configparser
import loadconfig
import matplotlib.pyplot as plt
import preprocessing_utils as pu
import define_models_torch
import time


config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

pretraining_dataset_path = cfg.get('pretraining', 'pretraining_dataset_path')
num_cnn_layers = cfg.getint('pretraining', 'num_cnn_layers')
preprocessing_ID = cfg.get('pretraining', 'preprocessing_ID')
sequence_length = cfg.getfloat('pretraining', 'sequence_length')
num_classes_librispeech = cfg.getint('pretraining', 'num_classes_librispeech')

#training parameters
experiment_name = '4secs_inv'
gpu_id = 0
num_epochs = 50
batch_size = 40
learning_rate = 0.00005
regularization_lambda = 0.
base_results = '../pretraining_vgg/librispeech'
save_model_metric = 'loss'
early_stopping = True
patience = 5
load_pretrained = False
sr = 16000

#compute correct pad dimension
dummy = np.zeros(int(sequence_length*sr))
seg_fft = pu.spectrum_fast(dummy)
x_dim, y_dim = seg_fft.shape

def dyn_pad(input, x_target=x_dim):
    #dynamic time zeropadding of 4d tensor in the 3rd (time) dim
    #because librispeech one-word dataset doesn't fit in ram if padded
    b,c,x,y = input.shape
    pad = torch.zeros(b,c,x_target,y)
    diff = x_target - x - 1
    random_init = np.random.randint(diff)
    pad[:,:,random_init:random_init+x,:] = input

    return pad

#output filenames
base_results = os.path.join(base_results, experiment_name)
if not os.path.exists(base_results):
    os.makedirs(base_results)
model_path = os.path.join(base_results, 'model')
results_path = os.path.join(base_results, 'results.npy')
figure_path = os.path.join(base_results, 'figure.png')


device = 'cuda:' + str(gpu_id)

#load datasets
print ('Loading dataset')
loading_start = time.perf_counter()

training_predictors = np.load(os.path.join(pretraining_dataset_path, preprocessing_ID + '_training_predictors.npy'), \
                              allow_pickle=True)
training_target = np.load(os.path.join(pretraining_dataset_path, preprocessing_ID + '_training_target.npy'), \
                              allow_pickle=True)
'''
training_predictors = np.load(os.path.join(pretraining_dataset_path, preprocessing_ID + '_test_predictors.npy'), \
                              allow_pickle=True)
training_target = np.load(os.path.join(pretraining_dataset_path, preprocessing_ID + '_test_target.npy'), \
                              allow_pickle=True)
'''
validation_predictors = np.load(os.path.join(pretraining_dataset_path, preprocessing_ID + '_validation_predictors.npy'), \
                              allow_pickle=True)
validation_target = np.load(os.path.join(pretraining_dataset_path, preprocessing_ID + '_validation_target.npy'), \
                              allow_pickle=True)
test_predictors = np.load(os.path.join(pretraining_dataset_path, preprocessing_ID + '_test_predictors.npy'), \
                              allow_pickle=True)
test_target = np.load(os.path.join(pretraining_dataset_path, preprocessing_ID + '_test_target.npy'), \
                              allow_pickle=True)

'''
bound = 100
training_predictors = training_predictors[:bound]
training_target = training_target[:bound]
validation_predictors = validation_predictors[:bound]
validation_target = validation_target[:bound]
test_predictors = test_predictors[:bound]
test_target = test_target[:bound]
'''
#normalize datasets
tr_mean = np.mean(training_predictors)
tr_std = np.std(training_predictors)
training_predictors = np.subtract(training_predictors, tr_mean)
training_predictors = np.divide(training_predictors, tr_std)
validation_predictors = np.subtract(validation_predictors, tr_mean)
validation_predictors = np.divide(validation_predictors, tr_std)
test_predictors = np.subtract(test_predictors, tr_mean)
test_predictors = np.divide(test_predictors, tr_std)

#reshaping for cnn
training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1],training_predictors.shape[2])
validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1], validation_predictors.shape[2])
test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1], test_predictors.shape[2])

#convert to tensor
train_predictors = torch.tensor(training_predictors).float()
val_predictors = torch.tensor(validation_predictors).float()
test_predictors = torch.tensor(test_predictors).float()
train_target = torch.tensor(training_target).long()
val_target = torch.tensor(validation_target).long()
test_target = torch.tensor(test_target).long()

#build dataset from tensors
tr_dataset = utils.TensorDataset(train_predictors, train_target)
val_dataset = utils.TensorDataset(val_predictors, val_target)
test_dataset = utils.TensorDataset(test_predictors, test_target)

#build data loader from dataset
tr_data = utils.DataLoader(tr_dataset, batch_size, shuffle=True, pin_memory=True)
val_data = utils.DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True)
test_data = utils.DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True)  #no batch here!!

#load model
par = ['output_classes=' + str(num_classes_librispeech)]

model, dummy = define_models_torch.vgg16(0,0, par)

model = model.to(device)

print (model)


#compute number of parameters
model_params = sum([np.prod(p.size()) for p in model.parameters()])
print ('Total paramters: ' + str(model_params))

#define optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                              weight_decay=regularization_lambda)
loss_function = nn.CrossEntropyLoss()

#init history
train_loss_hist = []
val_loss_hist = []
train_acc_hist = []
val_acc_hist = []


loading_time = float(time.perf_counter()) - float(loading_start)
print ('\nLoading time: ' + str(np.round(float(loading_time), decimals=1)) + ' seconds')


for epoch in range(num_epochs):
    epoch_start = time.perf_counter()
    model.train()
    print ('\n')
    string = 'Epoch: [' + str(epoch+1) + '/' + str(num_epochs) + '] '
    #iterate batches
    for i, (sounds, truth) in enumerate(tr_data):
        #sounds = torch.cat((sounds,sounds,sounds), axis=1)  #because vgg wants 3 channels as input
        sounds = dyn_pad(sounds)
        sounds = sounds.to(device)
        truth = truth.to(device)
        optimizer.zero_grad()
        outputs = model(sounds)
        loss = loss_function(outputs, truth)
        loss.backward()
        #print
        #print progress and update history, optimizer step
        perc = int(i / len(tr_data) * 20)
        inv_perc = int(20 - perc - 1)
        loss_print_t = str(np.round(loss.item(), decimals=5))
        string_progress = string + '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' loss: ' + loss_print_t
        print ('\r', string_progress, end='')
        optimizer.step()

        #validation loss, training and val accuracy computation
        #after current epoch training
        train_batch_losses = []
        val_batch_losses = []
        train_batch_accs = []
        val_batch_accs = []

    with torch.no_grad():
        model.eval()
        #training data
        for i, (sounds, truth) in enumerate(tr_data):
            optimizer.zero_grad()
            #sounds = torch.cat((sounds,sounds,sounds), axis=1)  #because vgg wants 3 channels as input
            sounds = dyn_pad(sounds)
            sounds = sounds.to(device)
            truth = truth.to(device)
            outputs = model(sounds)
            temp_loss = loss_function(outputs, truth)
            train_batch_losses.append(temp_loss.item())
            temp_acc = accuracy_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            train_batch_accs.append(temp_acc)

        #validation data
        for i, (sounds, truth) in enumerate(val_data):
            #sounds = torch.cat((sounds,sounds,sounds), axis=1)  #because vgg wants 3 channels as input
            optimizer.zero_grad()
            sounds = dyn_pad(sounds)
            sounds = sounds.to(device)
            truth = truth.to(device)
            outputs = model(sounds)
            temp_loss = loss_function(outputs, truth)
            val_batch_losses.append(temp_loss.item())
            temp_acc = accuracy_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            val_batch_accs.append(temp_acc)

    #append to history and print
    train_epoch_loss = np.mean(train_batch_losses)
    train_loss_hist.append(train_epoch_loss)
    val_epoch_loss = np.mean(val_batch_losses)
    val_loss_hist.append(val_epoch_loss)
    train_epoch_acc = np.mean(train_batch_accs)
    train_acc_hist.append(train_epoch_acc)
    val_epoch_acc = np.mean(val_batch_accs)
    val_acc_hist.append(val_epoch_acc)
    print ('\n  Train loss: ' + str(np.round(train_epoch_loss.item(), decimals=5)) + ' | Val loss: ' + str(np.round(val_epoch_loss.item(), decimals=5)))
    print ('  Train acc: ' + str(np.round(train_epoch_acc.item(), decimals=5)) + ' | Val acc: ' + str(np.round(val_epoch_acc.item(), decimals=5)))

    #compute epoch time
    epoch_time = float(time.perf_counter()) - float(epoch_start)
    print ('\n Epoch time: ' + str(np.round(float(epoch_time), decimals=1)) + ' seconds')

    #save best model (metrics = validation loss)
    if epoch == 0:
        torch.save(model.state_dict(), model_path)
        print ('\nModel saved')
        saved_epoch = epoch + 1
    else:
        if save_model_metric == 'loss':
            best_loss = min(val_loss_hist[:-1])  #not looking at curr_loss
            curr_loss = val_loss_hist[-1]
            if curr_loss < best_loss:
                torch.save(model.state_dict(), model_path)
                print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                saved_epoch = epoch + 1

        elif save_model_metric == 'acc':
            best_acc = max(val_acc_hist[:-1])  #not looking at curr_loss
            curr_acc = val_acc_hist[-1]
            if curr_acc > best_acc:
                torch.save(model.state_dict(), model_path)
                print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                saved_epoch = epoch + 1

        else:
            raise ValueError('Wrong metric selected')

    if early_stopping and epoch >= patience+1:
        patience_vec = val_loss_hist[-patience+1:]
        best_l = np.argmin(patience_vec)
        if best_l == 0:
            print ('Training early-stopped')
            break



train_batch_losses = []
val_batch_lesses = []
test_batch_losses = []
task_type = 'classification'

if task_type == 'classification':
    train_batch_accs = []
    val_batch_accs = []
    test_batch_accs = []

    train_batch_f1 = []
    val_batch_f1 = []
    test_batch_f1 = []

    train_batch_precision = []
    val_batch_precision = []
    test_batch_precision = []

    train_batch_recall = []
    val_batch_recall = []
    test_batch_recall = []

elif task_type == 'regression':
    train_batch_rmse = []
    val_batch_rmse = []
    test_batch_rmse = []

    train_batch_mae = []
    val_batch_mae = []
    test_batch_mae = []

model.eval()
with torch.no_grad():

    #TRAINING DATA
    for i, (sounds, truth) in enumerate(tr_data):
        optimizer.zero_grad()
        #sounds = torch.cat((sounds,sounds,sounds), axis=1)  #because vgg wants 3 channels as input
        sounds = dyn_pad(sounds)
        sounds = sounds.to(device)
        truth = truth.to(device)
        outputs = model(sounds)

        temp_loss = loss_function(outputs, truth)
        train_batch_losses.append(temp_loss.item())

        if task_type == 'classification':
            temp_acc = accuracy_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            train_batch_accs.append(temp_acc)
            temp_f1 = f1_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float(), average="macro")
            train_batch_f1.append(temp_f1)
            temp_precision = precision_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float(), average="macro")
            train_batch_precision.append(temp_precision)
            temp_recall = recall_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float(), average="macro")
            train_batch_recall.append(temp_recall)

        elif task_type == 'regression':
            temp_rmse = mean_squared_error(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            train_batch_rmse.append(temp_rmse)
            temp_mae = mean_absolute_error(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            train_batch_mae.append(temp_mae)


    #VALIDATION DATA
    for i, (sounds, truth) in enumerate(val_data):
        optimizer.zero_grad()
        #sounds = torch.cat((sounds,sounds,sounds), axis=1)  #because vgg wants 3 channels as input
        sounds = dyn_pad(sounds)
        sounds = sounds.to(device)
        truth = truth.to(device)
        outputs = model(sounds)

        temp_loss = loss_function(outputs, truth)
        val_batch_losses.append(temp_loss.item())

        if task_type == 'classification':
            temp_acc = accuracy_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            val_batch_accs.append(temp_acc)
            temp_f1 = f1_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float(), average="macro")
            val_batch_f1.append(temp_f1)
            temp_precision = precision_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float(), average="macro")
            val_batch_precision.append(temp_precision)
            temp_recall = recall_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float(), average="macro")
            val_batch_recall.append(temp_recall)

        elif task_type == 'regression':
            temp_rmse = mean_squared_error(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            val_batch_rmse.append(temp_rmse)
            temp_mae = mean_absolute_error(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            val_batch_mae.append(temp_mae)

    #TEST DATA
    for i, (sounds, truth) in enumerate(test_data):
        optimizer.zero_grad()
        #sounds = torch.cat((sounds,sounds,sounds), axis=1)  #because vgg wants 3 channels as input
        sounds = dyn_pad(sounds)
        sounds = sounds.to(device)
        truth = truth.to(device)
        outputs = model(sounds)

        temp_loss = loss_function(outputs, truth)
        test_batch_losses.append(temp_loss.item())

        if task_type == 'classification':
            temp_acc = accuracy_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            test_batch_accs.append(temp_acc)
            temp_f1 = f1_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float(), average="macro")
            test_batch_f1.append(temp_f1)
            temp_precision = precision_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float(), average="macro")
            test_batch_precision.append(temp_precision)
            temp_recall = recall_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float(), average="macro")
            test_batch_recall.append(temp_recall)

        elif task_type == 'regression':
            temp_rmse = mean_squared_error(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            test_batch_rmse.append(temp_rmse)
            temp_mae = mean_absolute_error(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
            test_batch_mae.append(temp_mae)

#save results in temp dict file
temp_results = {}

#save loss
temp_results['train_loss'] = np.mean(train_batch_losses)
temp_results['val_loss'] = np.mean(val_batch_losses)
temp_results['test_loss'] = np.mean(test_batch_losses)

#if classification compute also f1, precision, recall
if task_type == 'classification':
    temp_results['train_acc'] = np.mean(train_batch_accs)
    temp_results['val_acc'] = np.mean(val_batch_accs)
    temp_results['test_acc'] = np.mean(test_batch_accs)

    temp_results['train_f1'] = np.mean(train_batch_f1)
    temp_results['val_f1'] = np.mean(val_batch_f1)
    temp_results['test_f1'] = np.mean(test_batch_f1)

    temp_results['train_precision'] = np.mean(train_batch_precision)
    temp_results['val_precision'] = np.mean(val_batch_precision)
    temp_results['test_precision'] = np.mean(test_batch_precision)

    temp_results['train_recall'] = np.mean(train_batch_recall)
    temp_results['val_recall'] = np.mean(val_batch_recall)
    temp_results['test_recall'] = np.mean(test_batch_recall)
#save acc if classification append classification metrics
elif task_type == 'regression':
    temp_results['train_MAE'] = np.mean(train_batch_mae)
    temp_results['val_MAE'] = np.mean(val_batch_mae)
    temp_results['test_MAE'] = np.mean(test_batch_mae)

    temp_results['train_RMSE'] = np.mean(train_batch_rmse)
    temp_results['val_RMSE'] = np.mean(val_batch_rmse)
    temp_results['test_RMSE'] = np.mean(test_batch_rmse)

#save history
temp_results['train_loss_hist'] = train_loss_hist
temp_results['val_loss_hist'] = val_loss_hist
if task_type == 'classification':
    temp_results['train_acc_hist'] = train_acc_hist
    temp_results['val_acc_hist'] = val_acc_hist

plt.subplot(211)
plt.title('Loss History')
plt.plot(train_loss_hist)
plt.plot(val_loss_hist)
plt.legend(['train', 'val'])
plt.subplot(212)
plt.title('Acc History')
plt.plot(train_acc_hist)
plt.plot(val_acc_hist)
plt.legend(['train', 'val'])
plt.savefig(figure_path)


np.save(results_path, temp_results)

#print train results
print ('')
print ('\n train results:')
for i in temp_results.keys():
    if 'hist' not in i and 'actors' not in i:
        if 'train' in i:
            print (str(i) + ': ' + str(temp_results[i]))
print ('\n val results:')
for i in temp_results.keys():
    if 'hist' not in i and 'actors' not in i:
        if 'val' in i:
            print (str(i) + ': ' + str(temp_results[i]))
print ('\n test results:')
for i in temp_results.keys():
    if 'hist' not in i and 'actors' not in i:
        if 'test' in i:
            print (str(i) + ': ' + str(temp_results[i]))
