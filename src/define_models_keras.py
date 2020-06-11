from __future__ import print_function
import numpy as np
import configparser
import matplotlib.pyplot as plt
import keras.backend as k
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, Conv2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, Reshape, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.applications.resnet50 import ResNet50
from keras.models import load_model
from keras.utils import np_utils
from keras.backend import int_shape
from keras import regularizers
from keras import optimizers
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

def EXAMPLE_model_classification(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    to use this model, simply call architecture=EXAMPLE_model as a parameter
    in the UI script
    '''
    #FIRST, DECLARE DEFAULT PARAMETERS OF YOUR MODEL AS KEYS OF A DICT
    #default parameters
    p = {
    'regularization_lambda': 0.1,
    'kernel_size_1': [16, 12],
    'kernel_size_2': [8, 12],
    'kernel_size_3': [5,7],
    'pool_size': [2,2],
    'conv1_depth': 20,
    'conv2_depth': 28,
    'conv3_depth': 40,
    'drop_prob': 0.3,
    'hidden_size': 200}

    reg = regularizers.l2(p['regularization_lambda'])

    #THEN CALL PARSE_PAREMETERS TO OVERWRITE DEFAULT PARAMETERS
    #WITH PARAMETERS DEFINED IN THE UI SCRIPT
    p = parse_parameters(p, user_parameters)

    #FINALLY DECLARE YOUR ARCHITECTURE AND RETURN THE MODEL
    input_data = Input(shape=(time_dim, features_dim, 1))  #time_dim and features_dim are not from the dict
    conv_1 = Convolution2D(p['conv1_depth'], (p['kernel_size_1'][0],p['kernel_size_1'][1]), padding='same', activation='tanh')(input_data)
    pool_1 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_1)
    conv_2 = Convolution2D(p['conv2_depth'], (p['kernel_size_2'][0],p['kernel_size_2'][1]), padding='same', activation='tanh')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_2)
    conv_3 = Convolution2D(p['conv3_depth'], (p['kernel_size_3'][0],p['kernel_size_3'][1]), padding='same', activation='tanh')(pool_2)
    pool_3 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_3)
    flat = Flatten()(pool_3)
    drop_1 = Dropout(p['drop_prob'])(flat)
    hidden = Dense(p['hidden_size'], activation='tanh', kernel_regularizer=reg)(drop_1)
    out = Dense(8, activation='softmax')(hidden)
    model = Model(inputs=input_data, outputs=out)

    #always return model AND p!!!
    return model, p

def simple_CNN(time_dim, features_dim, user_parameters=['niente = 0']):
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
    'num_classes':8
    }


    #THEN CALL PARSE_PAREMETERS TO OVERWRITE DEFAULT PARAMETERS
    #WITH PARAMETERS DEFINED IN THE UI SCRIPT
    p = parse_parameters(p, user_parameters)
    reg = regularizers.l2(p['regularization_lambda'])

    #FINALLY DECLARE YOUR ARCHITECTURE AND RETURN THE MODEL
    input_data = Input(shape=(time_dim, features_dim, 1))  #time_dim and features_dim are not from the dict
    conv_1 = Convolution2D(p['channels'], (p['kernel_size_1'][0],p['kernel_size_1'][1]), padding='same', activation='tanh')(input_data)
    flat = Flatten()(conv_1)
    hidden = Dense(p['hidden_size'], activation='tanh', kernel_regularizer=reg)(flat)
    out = Dense(8, activation='softmax')(hidden)
    model = Model(inputs=input_data, outputs=out)

    #always return model AND p!!!
    return model, p

def OMG_model(time_dim, features_dim, user_parameters=['niente = 0']):

    p = {
    'regularization_lambda': 0.1,
    'kernel_size_1': [16, 12],
    'kernel_size_2': [8, 12],
    'kernel_size_3': [5,7],
    'pool_size': [2,2],
    'conv1_depth': 20,
    'conv2_depth': 28,
    'conv3_depth': 40,
    'drop_prob': 0.3,
    'hidden_size': 200,
    'output_classes': 8,
    'load_weights': False,
    'pretrained_model': 'model_xval_ravdess_cqt_exp1_run2_fold0'}

    reg = regularizers.l2(p['regularization_lambda'])

    p = parse_parameters(p, user_parameters)

    input_data = Input(shape=(time_dim, features_dim, 1))  #time_dim and features_dim are not from the dict

    conv_1 = Convolution2D(p['conv1_depth'], (p['kernel_size_1'][0],p['kernel_size_1'][1]), padding='same', activation='tanh')(input_data)
    pool_1 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_1)

    conv_2 = Convolution2D(p['conv2_depth'], (p['kernel_size_2'][0],p['kernel_size_2'][1]), padding='same', activation='tanh')(pool_1)
    pool_2 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_2)

    conv_3 = Convolution2D(p['conv3_depth'], (p['kernel_size_3'][0],p['kernel_size_3'][1]), padding='same', activation='tanh')(pool_2)
    pool_3 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_3)

    flat = Flatten()(pool_3)

    drop_1 = Dropout(p['drop_prob'])(flat)
    hidden = Dense(p['hidden_size'], activation='tanh', kernel_regularizer=reg)(drop_1)

    out = Dense(p['output_classes'], activation='softmax')(hidden)

    model = Model(inputs=input_data, outputs=out)


    #load pretrained weights if desired
    if p['load_weights']:
        model
        pretrained_path = os.path.join(PRETRAINED_MODELS_FOLDER, p['pretrained_model'])
        num_layers = len([layer.name for layer in model.layers])
        pretrained = load_model(pretrained_path)
        shared_weights = []
        for layer in range(num_layers): #cut output layer
            shared_weights.append(model.layers[layer].name)
            try:
                model.layers[layer].set_weights(pretrained.layers[layer].get_weights())
            except ValueError:
                print ('')
                print ('')
                print ('Warning: weights shared from: ' + str(model.layers[0].name) + ' to: ' + str(model.layers[layer-1].name))
                print ('')
                print ('')
                break
        p['layers_with_shared_weights'] = shared_weights

    return model, p

def AlexNet(time_dim, features_dim, user_parameters=['niente = 0']):

    p = {
    'regularization_lambda': 0.1,
    'kernel_size_1': [11,11],
    'kernel_size_2': [5,5],
    'kernel_size_3': [3,3],
    'kernel_size_4': [3,3],
    'kernel_size_5': [3,3],
    'depth_1': 96,
    'depth_2': 256,
    'depth_3': 384,
    'depth_4': 384,
    'depth_5': 256,
    'pool_size': [2,2],
    'hidden_size_1': 4096,
    'hidden_size_2': 4096,
    'hidden_size_3': 1000,
    'drop_prob': 0.4,
    'output_classes': 8,
    'load_weights': False,
    'pretrained_model': 'AlexNet_pretrained_ravdess.hdf5'
    }

    p = parse_parameters(p, user_parameters)

    input_data = Input(shape=(time_dim, features_dim, 1))  #time_dim and features_dim are not from the dict

    conv_1 = Conv2D(p['depth_1'], kernel_size=p['kernel_size_1'],strides=(4,4), padding='valid', activation='relu')(input_data)
    norm_1 = BatchNormalization()(conv_1)

    conv_2 = Conv2D(p['depth_2'], kernel_size=p['kernel_size_2'],strides=(1,1), padding='valid', activation='relu')(norm_1)
    pool_2 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_2)
    norm_2 = BatchNormalization()(pool_2)

    conv_3 = Conv2D(p['depth_3'], kernel_size=p['kernel_size_3'],strides=(1,1), padding='valid', activation='relu')(norm_2)
    norm_3 = BatchNormalization()(conv_3)

    conv_4 = Conv2D(p['depth_4'], kernel_size=p['kernel_size_4'],strides=(1,1), padding='valid', activation='relu')(norm_3)
    norm_4 = BatchNormalization()(conv_4)

    conv_5 = Conv2D(p['depth_5'], kernel_size=p['kernel_size_5'],strides=(1,1), padding='valid', activation='relu')(norm_4)
    pool_5 = MaxPooling2D(pool_size=(p['pool_size'][0],p['pool_size'][1]))(conv_5)
    norm_5 = BatchNormalization()(pool_5)

    flat = Flatten()(norm_5)

    hidden_1 = Dense(p['hidden_size_1'], activation='relu')(flat)
    drop_1 = Dropout(p['drop_prob'])(hidden_1)
    norm_6 = BatchNormalization()(drop_1)

    hidden_2 = Dense(p['hidden_size_2'], activation='relu')(norm_6)
    drop_2 = Dropout(p['drop_prob'])(hidden_2)
    norm_7 = BatchNormalization()(drop_2)

    hidden_3 = Dense(p['hidden_size_3'], activation='relu')(norm_7)
    drop_3 = Dropout(p['drop_prob'])(hidden_3)
    norm_8 = BatchNormalization()(drop_3)

    out = Dense(p['output_classes'], activation='softmax')(norm_8)

    model = Model(inputs=input_data, outputs=out)

    #load pretrained weights if desired
    if p['load_weights']:
        model
        pretrained_path = os.path.join(PRETRAINED_MODELS_FOLDER, p['pretrained_model'])
        num_layers = len([layer.name for layer in model.layers])
        pretrained = load_model(pretrained_path)
        shared_weights = []
        for layer in range(num_layers): #cut output layer
            shared_weights.append(model.layers[layer].name)
            try:
                model.layers[layer].set_weights(pretrained.layers[layer].get_weights())
            except ValueError:
                print ('')
                print ('')
                print ('Warning: weights shared from: ' + str(model.layers[0].name) + ' to: ' + str(model.layers[layer-1].name))
                print ('')
                print ('')
                break
        p['layers_with_shared_weights'] = shared_weights

    return model, p

def ParallelConv(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    https://pdfs.semanticscholar.org/810d/2a659a599572825d62dbabf28d233ce0d8b1.pdf
    '''
    # K*((W−F+2P)/S+1), where W - input volume size, F the receptive field size of the
    #Conv Layer neurons, S - the stride with which they are applied,
    #P - the amount of zero padding used on the border, K - the depth of conv layer.

    p = {
    'kernel_size_1': [12,16],
    'kernel_size_2': [18,24],
    'kernel_size_3': [24,32],
    'kernel_size_4': [30,40],
    'depth': 85,
    'hidden_size_1': 400 ,
    'hidden_size_2': 200,
    'drop_prob': 0.5,
    'p_divider': 2,
    'load_weights': False,
    'pretrained_model': 'ParallelConv_pretrained_ravdess.hdf5'
    }

    p = parse_parameters(p, user_parameters)

    #compute pooling parameters as in paper
    pool_size_1 = [int((time_dim-p['kernel_size_1'][0])/p['p_divider']), int((features_dim-p['kernel_size_1'][1])/p['p_divider'])]
    pool_size_2 = [int((time_dim-p['kernel_size_2'][0])/p['p_divider']), int((features_dim-p['kernel_size_2'][1])/p['p_divider'])]
    pool_size_3 = [int((time_dim-p['kernel_size_3'][0])/p['p_divider']), int((features_dim-p['kernel_size_3'][1])/p['p_divider'])]
    pool_size_4 = [int((time_dim-p['kernel_size_4'][0])/p['p_divider']), int((features_dim-p['kernel_size_4'][1])/p['p_divider'])]


    input_data = Input(shape=(time_dim, features_dim, 1))

    #parallel convs
    conv_1 = Conv2D(p['depth'], kernel_size=p['kernel_size_1'], activation='relu')(input_data)
    pool_1 = MaxPooling2D(pool_size=pool_size_1)(conv_1)
    flat_1 = Flatten()(pool_1)

    conv_2 = Conv2D(p['depth'], kernel_size=p['kernel_size_2'], activation='relu')(input_data)
    pool_2 = MaxPooling2D(pool_size=pool_size_2)(conv_2)
    flat_2 = Flatten()(pool_2)

    conv_3 = Conv2D(p['depth'], kernel_size=p['kernel_size_3'], activation='relu')(input_data)
    pool_3 = MaxPooling2D(pool_size=pool_size_3)(conv_3)
    flat_3 = Flatten()(pool_3)

    conv_4 = Conv2D(p['depth'], kernel_size=p['kernel_size_4'], activation='relu')(input_data)
    pool_4 = MaxPooling2D(pool_size=pool_size_4)(conv_4)
    flat_4 = Flatten()(pool_4)

    X = Concatenate()([flat_1,flat_2,flat_3,flat_4])

    drop_1 = Dropout(p['drop_prob'])(X)
    hidden_1 = Dense(p['hidden_size_1'], activation='relu')(drop_1)
    drop_2 = Dropout(p['drop_prob'])(hidden_1)
    norm_1 = BatchNormalization()(drop_2)

    hidden_2 = Dense(p['hidden_size_2'], activation='relu')(norm_1)
    norm_2 = BatchNormalization()(hidden_2)

    out = Dense(p['output_classes'], activation='softmax')(norm_2)

    model = Model(inputs=input_data, outputs=out)

    #load pretrained weights if desired
    if p['load_weights']:
        model
        pretrained_path = os.path.join(PRETRAINED_MODELS_FOLDER, p['pretrained_model'])
        num_layers = len([layer.name for layer in model.layers])
        pretrained = load_model(pretrained_path)
        shared_weights = []
        for layer in range(num_layers): #cut output layer
            shared_weights.append(model.layers[layer].name)
            try:
                model.layers[layer].set_weights(pretrained.layers[layer].get_weights())
            except ValueError:
                print ('')
                print ('')
                print ('Warning: weights shared from: ' + str(model.layers[0].name) + ' to: ' + str(model.layers[layer-1].name))
                print ('')
                print ('')
                break
        p['layers_with_shared_weights'] = shared_weights

    return model, p

def ResNet_50(time_dim, features_dim, user_parameters=['niente = 0']):
    '''
    standard resnet50
    '''
    p={
    'output_classes': 8
    }
    p = parse_parameters(p, user_parameters)

    input_data = Input(shape=(time_dim, features_dim, 1))


    base_model = ResNet50(include_top=False,input_tensor=input_data)

    # Add final layers
    x = base_model.output
    x = Flatten()(x)
    out = Dense(n_classes, activation='softmax', name='fc1000')(x)

    model = Model(input=base_model.input, output=out)

    return model, p
