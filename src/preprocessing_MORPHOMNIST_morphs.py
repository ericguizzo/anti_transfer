import numpy as np
import pandas as pd
import morphomnist.io as io
import morphomnist.measure as mmeasure
from skimage.transform import resize as resize
from scipy import interpolate
import matplotlib.pyplot as plt
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
labels_target = ['area','length','thickness','slant','width','height']
limit = 100
resize_dims = [32,32]

# arrays to compute range of each label
counts = {
'area' : [],
'length' : [],
'thickness' : [],
'slant' : [],
'width' : [],
'height' : []
}

predictors = []
target = []

#fill dataset arrays with selected labels
for set in ['train', 't10k']: #iterate sets (train and test)
    #load data
    ims_file = list(filter(lambda x: set in x and 'images' in x, contents))
    labels_file = list(filter(lambda x: set in x and 'labels' in x, contents))
    morpho_file = list(filter(lambda x: set in x and 'morpho' in x, contents))
    ims_path = os.path.join(input_morphomnist_folder, ims_file[0])
    labels_path = os.path.join(input_morphomnist_folder, labels_file[0])
    morpho_path = os.path.join(input_morphomnist_folder, morpho_file[0])
    ims = io.load_idx(ims_path)
    labels = io.load_idx(labels_path)
    morph_labels = pd.read_csv(morpho_path)

    n_data = ims.shape[0]
    limit = n_data
    for i in range(limit):  #iterate all datapoints
        target_array = []
        for l in labels_target:  #iterate all wanted labels
            value = morph_labels.iloc[i][l]
            target_array.append(value)
            counts[l].append(value)
        resized = resize(ims[i], resize_dims)
        expanded = np.expand_dims(resized, axis=0)
        predictors.append(expanded)
        target.append(target_array)
        #print (expanded.shape, np.array(target_array).shape)
    #print (len(predictors), len(target))

#normalize all label (one by one) within 0-1 interval
norms = {}
for l in labels_target:  #order is important here
    norms[l] = [np.min(counts[l]), np.max(counts[l])] #min and max of each label

norm_target = []
for i in target:  #iterate every data point
    temp_arr = []
    for l in range(len(labels_target)):  #iterate every label
        curr_l = labels_target[l]
        curr_min = norms[curr_l][0]
        curr_max = norms[curr_l][1]
        curr_val = i[l]
        norm_val = (curr_val - curr_min) / (curr_max - curr_min)
        temp_arr.append(norm_val)
    norm_target.append(np.expand_dims(temp_arr,axis=0))

print ('culo', np.array(target).shape)
'''
area_x = np.zeros(limit)
length_x = np.ones(limit)
thickness_x = np.ones(limit) * 2
slant_x = np.ones(limit) * 3
width_x = np.ones(limit) * 4
height_x = np.ones(limit) * 5
#plt.scatter(area_x, counts['area'])
plt.figure(1)
plt.plot(counts['area'])
plt.title('area')
plt.figure(2)
plt.plot(counts['length'])
plt.title('length')
plt.figure(3)
plt.plot(counts['thickness'])
plt.title('thickness')
plt.figure(4)
plt.plot(counts['slant'])
plt.title('width')
plt.figure(5)
plt.plot(counts['height'])
plt.title('height')
plt.show()
'''
print (np.min(target), np.max(target))
print (np.min(norm_target), np.max(norm_target))

dict_predictors = {}
dict_target = {}

for i in range(len(predictors)):
    dict_predictors[i] = predictors[i]
    dict_target[i] = norm_target[i]

#save dicts
if not os.path.exists(os.path.join(OUTPUT_FOLDER, 'morphomnist_morpholabel')):
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'morphomnist_morpholabel'))


predictors_save_path = os.path.join(OUTPUT_FOLDER, 'morphomnist_morpholabel', 'morphomnist_morpholabel_predictors.npy')
target_save_path = os.path.join(OUTPUT_FOLDER, 'morphomnist_morpholabel', 'morphomnist_morpholabel_target.npy')

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
