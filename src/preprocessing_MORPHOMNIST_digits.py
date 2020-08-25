import numpy as np
import pandas as pd
import morphomnist.io as io
import morphomnist.measure as mmeasure
from skimage.transform import resize as resize
from scipy import interpolate
import matplotlib.pyplot as plt
import utility_functions as uf
import os, sys
import configparser
import loadconfig
'''
Preprocessing script.
Outputs numpy dicts containing preprocessed predictors and target
'''

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')

#input_morphomnist_folder = '/Users/eric/Downloads/global'
input_morphomnist_folder = cfg.get('preprocessing', 'input_morphomnist_folder')
contents = os.listdir(input_morphomnist_folder)

#all_labels = ['area','length','thickness','slant','width','height']
limit = 100
resize_dims = [32,32]

# arrays to compute range of each label

predictors = []
target = []

#fill dataset arrays with selected labels
for set in ['train', 't10k']: #iterate sets (train and test)
    #load data
    ims_file = list(filter(lambda x: set in x and 'images' in x, contents))
    labels_file = list(filter(lambda x: set in x and 'labels' in x, contents))
    ims_path = os.path.join(input_morphomnist_folder, ims_file[0])
    labels_path = os.path.join(input_morphomnist_folder, labels_file[0])
    ims = io.load_idx(ims_path)
    labels = io.load_idx(labels_path)

    n_data = ims.shape[0]
    limit = n_data
    for i in range(limit):  #iterate all datapoints
        label = uf.onehot(labels[i], 10)
        label = np.expand_dims(label, axis=0)
        resized = resize(ims[i], resize_dims)
        expanded = np.expand_dims(resized, axis=0)
        predictors.append(expanded)
        target.append(label)
        '''
        print (labels[i])
        plt.pcolormesh(resized)
        plt.show()
        plt.pause(2)
        '''
dict_predictors = {}
dict_target = {}

for i in range(len(predictors)):
    dict_predictors[i] = predictors[i]
    dict_target[i] = target[i]

#save dicts
predictors_save_path = os.path.join(OUTPUT_FOLDER, 'morphomnist_digits_predictors.npy')
target_save_path = os.path.join(OUTPUT_FOLDER, 'morphomnist_digits_target.npy')

print ('\nSaving matrices...')
np.save(predictors_save_path, dict_predictors)
np.save(target_save_path, dict_target)

#print dimensions
count = 0
predictors_dims = 0
keys = list(dict_predictors.keys())
for i in keys:
    count += dict_predictors[i].shape[0]
pred_shape = np.array(dict_predictors[keys[0]]).shape[1:]
tg_shape = np.array(dict_target[keys[0]]).shape[1:]
print ('')
print ('MATRICES SUCCESFULLY COMPUTED')
print ('')
print ('Total number of datapoints: ' + str(count))
print (' Predictors shape: ' + str(pred_shape))
print (' Target shape: ' + str(tg_shape))
