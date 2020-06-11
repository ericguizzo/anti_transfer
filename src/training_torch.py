#from __future__ import print_function
import sys, os
import time
#look at sys argv: if in crossvalidation model i/o matrices and new model filename
#are given from crossvalidation script, otherwise are normally taken from config.ini
try:
    cross_tag =  sys.argv[1]
    if cross_tag == 'crossvalidation':
        num_experiment = sys.argv[2]
        num_run = sys.argv[3]
        num_fold = sys.argv[4]
        parameters = sys.argv[5]
        model_path = sys.argv[6]
        results_path = sys.argv[7]
        output_temp_data_path = sys.argv[8]
        dataset = sys.argv[9]
        gpu_ID = int(sys.argv[10])
        num_folds = int(sys.argv[11])
        task_type = sys.argv[12]
        parameters_path = sys.argv[13]
        task_type = sys.argv[14]
        generator = eval(sys.argv[15])
        SAVE_MODEL = model_path

        print('crossvalidation mode: I/O from crossvalidation script')
        print('')
        print ('dataset: ' + dataset)
        print ('')
        print ('saving results at: ' + results_path)
        print('saving model at: ' + SAVE_MODEL + '.hdf5')
        print ('')

except IndexError:
    #test parameters
    #IF IN TEST MODE:no xvalidation, results saved as exp0
    #generator: 11865
    #nogenerator
    generator = True
    dataset = 'iemocap_randsplit_spectrum_fast'
    architecture = 'vgg16'
    parameters = ['niente = 0', "output_classes=4"]
    task_type = 'classification'
    SAVE_MODEL = '../models/prova'
    results_path = '../results/provisional'
    parameters_path = results_path + '/parameters'
    SAVE_RESULTS = results_path
    num_fold = 0
    num_exp = 0
    num_experiment = 0
    num_run = 0
    num_folds = 1
    gpu_ID = 1

    print ('test mode: I/O from config.ini file')
    print ('')
    print ('dataset: ' + dataset)
    print ('')
    print ('saving results at: ' + SAVE_RESULTS)
    print ('')
    print ('saving model at: ' + SAVE_MODEL + '.hdf5')
    print ('')



import loadconfig
import configparser
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.utils.data as utils
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, mean_absolute_error
import numpy as np
from multiscale_convlayer2 import MultiscaleConv2d
import utility_functions as uf
import define_models_torch as choose_model
import feature_loss


#import preprocessing_DAIC as pre

#np.random.seed(0)
#torch.manual_seed(0)
print('')
config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#load parameters from config file only test mode
DATASET_FOLDER = cfg.get('preprocessing', 'output_folder')

predictors_name = dataset + '_predictors.npy'
target_name = dataset + '_target.npy'
PREDICTORS_LOAD = os.path.join(DATASET_FOLDER, predictors_name)
TARGET_LOAD = os.path.join(DATASET_FOLDER, target_name)

#default training parameters
train_split = cfg.getfloat('training_defaults', 'train_split')
validation_split = cfg.getfloat('training_defaults', 'validation_split')
test_split = cfg.getfloat('training_defaults', 'test_split')
shuffle_training_data = eval(cfg.get('training_defaults', 'shuffle_training_data'))
save_best_model_metric = cfg.get('training_defaults', 'save_best_model_metric')
save_best_model_mode = cfg.get('training_defaults', 'save_best_model_mode')
early_stopping = eval(cfg.get('training_defaults', 'early_stopping'))
patience = cfg.getint('training_defaults', 'patience')
batch_size = cfg.getint('training_defaults', 'batch_size')
num_epochs = cfg.getint('training_defaults', 'num_epochs')
learning_rate = cfg.getfloat('training_defaults', 'learning_rate')
reshaping_type = cfg.get('training_defaults', 'reshaping_type')
choose_optimizer = cfg.get('training_defaults', 'choose_optimizer')

recompute_matrices = eval(cfg.get('training_defaults', 'recompute_matrices'))
regularization_lambda = cfg.getfloat('training_defaults', 'regularization_lambda')

#feature loss params
anti_transfer = eval(cfg.get('training_defaults', 'anti_transfer'))
at_pretraining = eval(cfg.get('training_defaults', 'at_pretraining'))
at_beta = cfg.getfloat('training_defaults', 'at_beta')
at_layer = cfg.getint('training_defaults', 'at_layer')
at_aggregation = cfg.get('training_defaults', 'at_aggregation')
at_distance = cfg.get('training_defaults', 'at_distance')

at_librispeech_model_path = cfg.get('training_defaults', 'at_librispeech_model_path')
at_iemocap_model_path = cfg.get('training_defaults', 'at_iemocap_model_path')
at_nsynth_model_path = cfg.get('training_defaults', 'at_nsynth_model_path')
at_goodsounds_model_path = cfg.get('training_defaults', 'at_goodsounds_model_path')

pretraining_classes_librispeech = cfg.get('training_defaults', 'pretraining_classes_librispeech')
pretraining_classes_iemocap = cfg.get('training_defaults', 'pretraining_classes_iemocap')
pretraining_classes_nsynth = cfg.get('training_defaults', 'pretraining_classes_nsynth')
pretraining_classes_goodsounds = cfg.get('training_defaults', 'pretraining_classes_goodsounds')


percs = [train_split, validation_split, test_split]

device = torch.device('cuda:' + str(gpu_ID))

if task_type == 'classification':
    loss_function = nn.CrossEntropyLoss()
    metrics_list = ['accuracy']
elif task_type == 'regression':
    loss_function = nn.MSELoss()
else:
    raise ValueError('task_type can be only: multilabel_classification, binary_classification or regression')

#path for saving best val loss and best val acc models
BVL_model_path = SAVE_MODEL
at_dataset = 'culo'

#OVERWRITE DEFAULT PARAMETERS IF IN XVAL MODE
try:
    a = sys.argv[5]
    parameters = parameters.split('/')
    for param in parameters:
        #print (param)
        exec(param)

except IndexError:
    pass

#load pretrained_vgg for anti transfer learning
at_model_url = 'n'
at_model_classes = 0
if anti_transfer:
    print ('AT DATASET!!!!!!!!!!!!!!!!!', at_dataset)
    if at_dataset == 'librispeech':
        at_model_url = at_librispeech_model_path
        at_model_classes = pretraining_classes_librispeech
    if at_dataset == 'iemocap':
        at_model_url = at_iemocap_model_path
        at_model_classes = pretraining_classes_iemocap
    if at_dataset == 'nsynth':
        at_model_url = at_nsynth_model_path
        at_model_classes = pretraining_classes_nsynth
    if at_dataset == 'goodsounds':
        at_model_url = at_goodsounds_model_path
        at_model_classes = pretraining_classes_goodsounds

    pretrained_vgg = feature_loss.load_feature_extractor(gpu_ID, at_model_url, at_model_classes)
else:
    pretrained_vgg = 'nothing'



#build dict with all UPDATED training parameters
training_parameters = {'train_split': train_split,
    'validation_split': validation_split,
    'test_split': test_split,
    'shuffle_training_data': shuffle_training_data,
    'save_best_model_metric': save_best_model_metric,
    'save_best_model_mode': save_best_model_mode,
    'early_stopping': early_stopping,
    'patience': patience,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'learning_rate': learning_rate,
    'optimizer': choose_optimizer,
    'regularization_lambda': regularization_lambda,
    'anti_transfer': anti_transfer,
    'at_beta': at_beta,
    'at_aggregation': at_aggregation,
    'at_distance': at_distance
    }

if anti_transfer:
    if not isinstance(at_layer, list):
        at_layer = [at_layer]

def main():
    #CREATE DATASET
    #load numpy data
    print('\n loading dataset...')

    folds_dataset_path = '../dataset/matrices'
    curr_fold_string = dataset + '_test_target_fold_' + str(num_fold) + '.npy'
    curr_fold_path = os.path.join(folds_dataset_path, curr_fold_string)

    train_pred_path = dataset + '_training_predictors_fold_' + str(num_fold) + '.npy'
    train_target_path = dataset + '_training_target_fold_' + str(num_fold) + '.npy'
    train_pred_path = os.path.join(folds_dataset_path, train_pred_path)
    train_target_path = os.path.join(folds_dataset_path, train_target_path)

    val_pred_path = dataset + '_validation_predictors_fold_' + str(num_fold) + '.npy'
    val_target_path = dataset + '_validation_target_fold_' + str(num_fold) + '.npy'
    val_pred_path = os.path.join(folds_dataset_path, val_pred_path)
    val_target_path = os.path.join(folds_dataset_path, val_target_path)

    test_pred_path = dataset + '_test_predictors_fold_' + str(num_fold) + '.npy'
    test_target_path = dataset + '_test_target_fold_' + str(num_fold) + '.npy'
    test_pred_path = os.path.join(folds_dataset_path, test_pred_path)
    test_target_path = os.path.join(folds_dataset_path, test_target_path)

    #compute which actors put in train, val, test for current fold
    dummy = np.load(TARGET_LOAD,allow_pickle=True)
    dummy = dummy.item()
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #JUST WRITE A FUNCTION TO RE-ORDER foldable_list TO SPLIT
    #TRAIN/VAL/TEST IN A BALANCED WAY
    foldable_list = list(dummy.keys())
    fold_actors_list = uf.folds_generator(num_folds, foldable_list, percs)
    train_list = fold_actors_list[int(num_fold)]['train']
    val_list = fold_actors_list[int(num_fold)]['val']
    test_list = fold_actors_list[int(num_fold)]['test']
    del dummy

    #if tensors of current fold has not been computed:
    if recompute_matrices:
        predictors_merged = np.load(PREDICTORS_LOAD,allow_pickle=True)
        target_merged = np.load(TARGET_LOAD,allow_pickle=True)
        predictors_merged = predictors_merged.item()
        target_merged = target_merged.item()

        print ('\n building dataset for current fold')
        print ('\n training:')
        training_predictors, training_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, train_list)
        print ('\n validation:')

        validation_predictors, validation_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, val_list)
        print ('\n test:')
        test_predictors, test_target = uf.build_matrix_dataset(predictors_merged,
                                                            target_merged, test_list)

        np.save(train_pred_path, training_predictors)
        np.save(train_target_path, training_target)
        np.save(val_pred_path, validation_predictors)
        np.save(val_target_path, validation_target)
        np.save(test_pred_path, test_predictors)
        np.save(test_target_path, test_target)

    if not recompute_matrices:
        if not os.path.exists(test_target_path):
            #load merged dataset, compute and save current tensors
            predictors_merged = np.load(PREDICTORS_LOAD,allow_pickle=True)
            target_merged = np.load(TARGET_LOAD,allow_pickle=True)
            predictors_merged = predictors_merged.item()
            target_merged = target_merged.item()

            print ('\n building dataset for current fold')
            print ('\n training:')
            training_predictors, training_target = uf.build_matrix_dataset(predictors_merged,
                                                                target_merged, train_list)
            print ('\n validation:')

            validation_predictors, validation_target = uf.build_matrix_dataset(predictors_merged,
                                                                target_merged, val_list)
            print ('\n test:')
            test_predictors, test_target = uf.build_matrix_dataset(predictors_merged,
                                                                target_merged, test_list)

            np.save(train_pred_path, training_predictors)
            np.save(train_target_path, training_target)
            np.save(val_pred_path, validation_predictors)
            np.save(val_target_path, validation_target)
            np.save(test_pred_path, test_predictors)
            np.save(test_target_path, test_target)

        else:
            #load pre-computed tensors
            training_predictors = np.load(train_pred_path,allow_pickle=True)
            training_target = np.load(train_target_path,allow_pickle=True)
            validation_predictors = np.load(val_pred_path,allow_pickle=True)
            validation_target = np.load(val_target_path,allow_pickle=True)
            test_predictors = np.load(test_pred_path,allow_pickle=True)
            test_target = np.load(test_target_path,allow_pickle=True)

        #normalize to 0 mean and unity std (according to training set mean and std)
        tr_mean = np.mean(training_predictors)
        tr_std = np.std(training_predictors)
        training_predictors = np.subtract(training_predictors, tr_mean)
        training_predictors = np.divide(training_predictors, tr_std)
        validation_predictors = np.subtract(validation_predictors, tr_mean)
        validation_predictors = np.divide(validation_predictors, tr_std)
        test_predictors = np.subtract(test_predictors, tr_mean)
        test_predictors = np.divide(test_predictors, tr_std)

    #from onehot to float (CrossEntropyLoss requires this)
    if task_type == 'classification':
        tr_target = []
        v_target = []
        ts_target = []
        for i in training_target:
            tr_target.append(np.argmax(i))
        for i in validation_target:
            v_target.append(np.argmax(i))
        for i in test_target:
            ts_target.append(np.argmax(i))
        training_target = np.array(tr_target)
        validation_target = np.array(v_target)
        test_target = np.array(ts_target)

    #reshape tensors
    #INSERT HERE FUNCTION FOR CUSTOM RESHAPING!!!!!

    if reshaping_type == 'cnn':
        training_predictors = training_predictors.reshape(training_predictors.shape[0], 1, training_predictors.shape[1],training_predictors.shape[2])
        validation_predictors = validation_predictors.reshape(validation_predictors.shape[0], 1, validation_predictors.shape[1], validation_predictors.shape[2])
        test_predictors = test_predictors.reshape(test_predictors.shape[0], 1, test_predictors.shape[1], test_predictors.shape[2])

    else:
        raise ValueError('wrong reshaping type')

    #convert to tensor
    train_predictors = torch.tensor(training_predictors).float()
    val_predictors = torch.tensor(validation_predictors).float()
    test_predictors = torch.tensor(test_predictors).float()
    train_target = torch.tensor(training_target).long()
    val_target = torch.tensor(validation_target).long()
    test_target = torch.tensor(test_target).long()

    #build dataset from tensors
    #target i == predictors because autoencoding
    tr_dataset = utils.TensorDataset(train_predictors, train_target)
    val_dataset = utils.TensorDataset(val_predictors, val_target)
    test_dataset = utils.TensorDataset(test_predictors, test_target)

    #build data loader from dataset
    tr_data = utils.DataLoader(tr_dataset, batch_size, shuffle=True, pin_memory=True)
    val_data = utils.DataLoader(val_dataset, batch_size, shuffle=False, pin_memory=True)
    test_data = utils.DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True)  #no batch here!!
    #DNN input shape
    time_dim = training_predictors.shape[-2]
    features_dim = training_predictors.shape[-1]


    #load model
    model_string = 'model_class, model_parameters = choose_model.' + architecture + '(time_dim, features_dim, parameters)'
    exec(model_string)
    model = locals()['model_class']
    #load pretrained weights if desired
    if at_pretraining:
        if at_pretraining == 'librispeech':
            pretrained_path = '../pretraining_vgg/librispeech/4secs_inv/model'
        elif at_pretraining == 'iemocap':
            pretrained_path = '../pretraining_vgg/iemocap/first/model'
        elif at_pretraining == 'nsynth':
            pretrained_path = '../pretraining_vgg/nsynth/6secs/model'
        elif at_pretraining == 'goodsounds':
            pretrained_path = '../pretraining_vgg/goodsounds/first/model'
        print ('PRETRAINING!±!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print ('pretraining on: ', pretrained_path)
        model.features.load_state_dict(torch.load(pretrained_path,
                                    map_location=lambda storage, location: storage),
                                    strict=False)

    model = model.to(device)
    #print summary
    #summary(model, input_size=(1, time_dim, features_dim))

    #compute number of parameters
    model_params = sum([np.prod(p.size()) for p in model.parameters()])
    print ('Total paramters: ' + str(model_params))

    #define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=regularization_lambda)








    #run training
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    model_folder = os.path.dirname(SAVE_MODEL)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    train_loss_hist = []
    val_loss_hist = []
    if task_type == 'classification':
        train_acc_hist = []
        val_acc_hist = []
    if anti_transfer:
        train_feature_loss_hist = []
        val_feature_loss_hist = []


    #finally, TRAINING LOOP
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        model.train()
        print ('\n')
        string = 'Epoch: [' + str(epoch+1) + '/' + str(num_epochs) + '] '
        #iterate batches
        for i, (sounds, truth) in enumerate(tr_data):
            sounds = sounds.to(device)
            truth = truth.to(device)

            optimizer.zero_grad()
            outputs = model(sounds)
            loss = loss_function(outputs, truth)
            #add anti-transfer component of loss
            if anti_transfer:
                for curr_layer in at_layer:
                    anti_transfer_loss = feature_loss.feature_loss(sounds, model,
                                         pretrained_vgg, at_beta, curr_layer,
                                         at_aggregation, at_distance)
                    loss = loss + anti_transfer_loss
            loss.backward()
            #print
            #print progress and update history, optimizer step
            perc = int(i / len(tr_data) * 20)
            inv_perc = int(20 - perc - 1)

            loss_print_t = str(np.round(loss.item(), decimals=5))
            if anti_transfer:
                antiloss_print_t = str(anti_transfer_loss.item())
                string_progress = string + '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' loss: ' + loss_print_t + '| antiloss: ' + antiloss_print_t
            else:
                string_progress = string + '[' + '=' * perc + '>' + '.' * inv_perc + ']' + ' loss: ' + loss_print_t
            print ('\r', string_progress, end='')
            optimizer.step()

        #validation loss, training and val accuracy computation
        #after current epoch training
        train_batch_losses = []
        val_batch_losses = []
        if anti_transfer:
            train_batch_feature_losses = []
            val_batch_feature_losses = []

        if task_type == 'classification':
            train_batch_accs = []
            val_batch_accs = []

        with torch.no_grad():
            model.eval()
            #training data
            for i, (sounds, truth) in enumerate(tr_data):
                optimizer.zero_grad()
                sounds = sounds.to(device)
                truth = truth.to(device)

                outputs = model(sounds)
                temp_loss = loss_function(outputs, truth)

                if anti_transfer:
                    for curr_layer in at_layer:
                        anti_transfer_loss = feature_loss.feature_loss(sounds, model,
                                             pretrained_vgg, at_beta, curr_layer,
                                             at_aggregation, at_distance)
                        temp_loss = temp_loss + anti_transfer_loss
                    train_batch_feature_losses.append(anti_transfer_loss.item())
                train_batch_losses.append(temp_loss.item())

                if task_type == 'classification':
                    temp_acc = accuracy_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
                    train_batch_accs.append(temp_acc)

            #validation data
            for i, (sounds, truth) in enumerate(val_data):
                optimizer.zero_grad()
                sounds = sounds.to(device)
                truth = truth.to(device)

                outputs = model(sounds)
                temp_loss = loss_function(outputs, truth)

                if anti_transfer:
                    for curr_layer in at_layer:
                        anti_transfer_loss = feature_loss.feature_loss(sounds, model,
                                             pretrained_vgg, at_beta, curr_layer,
                                             at_aggregation, at_distance)
                        temp_loss = temp_loss + anti_transfer_loss
                    val_batch_feature_losses.append(anti_transfer_loss.item())
                val_batch_losses.append(temp_loss.item())

                if task_type == 'classification':
                    temp_acc = accuracy_score(np.argmax(outputs.cpu().float(), axis=1), truth.cpu().float())
                    val_batch_accs.append(temp_acc)


        #append to history and print
        train_epoch_loss = np.mean(train_batch_losses)
        train_loss_hist.append(train_epoch_loss)
        val_epoch_loss = np.mean(val_batch_losses)
        val_loss_hist.append(val_epoch_loss)
        if anti_transfer:
            train_epoch_feature_loss = np.mean(train_batch_feature_losses)
            val_epoch_feature_loss = np.mean(val_batch_feature_losses)
            train_feature_loss_hist.append(train_epoch_feature_loss)
            val_feature_loss_hist.append(val_epoch_feature_loss)
        if task_type == 'classification':
            train_epoch_acc = np.mean(train_batch_accs)
            train_acc_hist.append(train_epoch_acc)
            val_epoch_acc = np.mean(val_batch_accs)
            val_acc_hist.append(val_epoch_acc)


        epoch_time = float(time.perf_counter()) - float(epoch_start)
        print ('\n Epoch time: ' + str(np.round(float(epoch_time), decimals=1)) + ' seconds')

        #save best model (metrics = validation loss)
        if epoch == 0:
            torch.save(model.state_dict(), BVL_model_path)
            print ('\nModel saved')
            saved_epoch = epoch + 1
        else:
            if save_model_metric == 'loss':
                best_loss = min(val_loss_hist[:-1])  #not looking at curr_loss
                curr_loss = val_loss_hist[-1]
                if curr_loss < best_loss:
                    torch.save(model.state_dict(), BVL_model_path)
                    print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                    saved_epoch = epoch + 1

            elif save_model_metric == 'acc':
                best_acc = max(val_acc_hist[:-1])  #not looking at curr_loss
                curr_acc = val_acc_hist[-1]
                if curr_acc > best_acc:
                    torch.save(model.state_dict(), BVL_model_path)
                    print ('\nModel saved')  #SUBSTITUTE WITH SAVE MODEL FUNC
                    saved_epoch = epoch + 1

            else:
                raise ValueError('Wrong metric selected')

        if anti_transfer:
            print ('\n  FEATURE LOSS: TRAIN: ' + str(np.round(train_epoch_feature_loss.item(), decimals=5)) + ' | VAL: ' + str(np.round(val_epoch_feature_loss.item(), decimals=5)))

        print ('\n  Train loss: ' + str(np.round(train_epoch_loss.item(), decimals=5)) + ' | Val loss: ' + str(np.round(val_epoch_loss.item(), decimals=5)))
        if task_type == 'classification':
            print ('  Train acc: ' + str(np.round(train_epoch_acc.item(), decimals=5)) + ' | Val acc: ' + str(np.round(val_epoch_acc.item(), decimals=5)))


        utilstring = 'dataset: ' + str(dataset) + ', exp: ' + str(num_experiment) + ', run: ' + str(num_run) + ', fold: ' + str(num_fold)
        print ('')
        print (utilstring)
        #AS LAST THING, AFTER OPTIMIZER.STEP AND EVENTUAL MODEL SAVING
        #AVERAGE MULTISCALE CONV KERNELS!!!!!!!!!!!!!!!!!!!!!!!!!
        for layer in model.modules():
            if isinstance(layer, MultiscaleConv2d):
                layer.update_kernels()

        if early_stopping and epoch >= patience+1:
            patience_vec = val_loss_hist[-patience+1:]
            best_l = np.argmin(patience_vec)
            if best_l == 0:
                print ('Training early-stopped')
                break

        #END OF EPOCH LOOP

    #compute results on the best saved model
    torch.cuda.empty_cache()  #free GPU
    #load best saved model
    model.load_state_dict(torch.load(BVL_model_path), strict=False)

    #if anti_transfer:
    #    pretrained_vgg = feature_loss.load_feature_extractor(gpu_ID)


    train_batch_losses = []
    val_batch_lesses = []
    test_batch_losses = []

    #if there is any multiscale layer
    there_is_multiconv = False
    for layer in model.modules():
        if isinstance(layer, MultiscaleConv2d):
            there_is_multiconv = True

    if there_is_multiconv:
        train_batch_stretch_percs = []
        val_batch_stretch_percs= []
        test_batch_stretch_percs = []


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
            sounds = sounds.to(device)
            truth = truth.to(device)
            outputs = model(sounds)

            temp_loss = loss_function(outputs, truth)
            if anti_transfer:
                for curr_layer in at_layer:
                    anti_transfer_loss = feature_loss.feature_loss(sounds, model,
                                         pretrained_vgg, at_beta, curr_layer,
                                         at_aggregation, at_distance)
                    temp_loss = temp_loss + anti_transfer_loss
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

            for layer in model.modules():
                if isinstance(layer, MultiscaleConv2d):
                    temp_stretch_percs = layer.get_stretch_percs()
                    train_batch_stretch_percs.append(temp_stretch_percs)


        #VALIDATION DATA
        for i, (sounds, truth) in enumerate(val_data):
            optimizer.zero_grad()
            sounds = sounds.to(device)
            truth = truth.to(device)
            outputs = model(sounds)

            temp_loss = loss_function(outputs, truth)
            if anti_transfer:
                for curr_layer in at_layer:
                    anti_transfer_loss = feature_loss.feature_loss(sounds, model,
                                         pretrained_vgg, at_beta, curr_layer,
                                         at_aggregation, at_distance)
                    temp_loss = temp_loss + anti_transfer_loss
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

            for layer in model.modules():
                if isinstance(layer, MultiscaleConv2d):
                    temp_stretch_percs = layer.get_stretch_percs()
                    val_batch_stretch_percs.append(temp_stretch_percs)

        #TEST DATA
        for i, (sounds, truth) in enumerate(test_data):
            optimizer.zero_grad()
            sounds = sounds.to(device)
            truth = truth.to(device)
            outputs = model(sounds)

            temp_loss = loss_function(outputs, truth)
            if anti_transfer:
                for curr_layer in at_layer:
                    anti_transfer_loss = feature_loss.feature_loss(sounds, model,
                                         pretrained_vgg, at_beta, curr_layer,
                                         at_aggregation, at_distance)
                    temp_loss = temp_loss + anti_transfer_loss
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

            for layer in model.modules():
                if isinstance(layer, MultiscaleConv2d):
                    temp_stretch_percs = layer.get_stretch_percs()
                    test_batch_stretch_percs.append(temp_stretch_percs)
                    print (temp_stretch_percs)


    #save results in temp dict file
    temp_results = {}

    #save loss
    temp_results['train_loss'] = np.mean(train_batch_losses)
    temp_results['val_loss'] = np.mean(val_batch_losses)
    temp_results['test_loss'] = np.mean(test_batch_losses)

    #save stretch percs if multiconv
    if there_is_multiconv:
        temp_results['train_stretch_percs'] = np.mean(train_batch_stretch_percs, axis=0)
        temp_results['val_stretch_percs'] = np.mean(val_batch_stretch_percs, axis=0)
        temp_results['test_stretch_percs'] = np.mean(test_batch_stretch_percs, axis=0)

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
    if anti_transfer:
        temp_results['train_feature_loss_hist'] = train_feature_loss_hist
        temp_results['val_feature_loss_hist'] = val_feature_loss_hist

    #save actors present in current fold
    temp_results['training_actors'] = train_list
    temp_results['validation_actors'] = val_list
    temp_results['test_actors'] = test_list

    #save parameters dict
    for i in training_parameters.keys():
        if i in locals()['model_parameters'].keys():
            del locals()['model_parameters'][i]

    with open(parameters_path, 'w') as f:
        f.write('%s\n' % ('TRAINING PARAMETERS:'))
        for key, value in training_parameters.items():
            f.write('%s:%s\n' % (key, value))
        f.write('%s\n' % (''))
        f.write('%s\n' % ('MODEL PARAMETERS:'))
        for key, value in locals()['model_parameters'].items():
            f.write('%s:%s\n' % (key, value))

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

if __name__ == '__main__':
    main()
