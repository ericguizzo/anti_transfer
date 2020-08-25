import numpy as np
from shutil import copy
import random
import os
import configparser
import loadconfig

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
in_path = cfg.get('preprocessing', 'input_gsc_raw_folder')
out_path = cfg.get('preprocessing', 'balanced_gsc_folder')

#in_path = '../../shared_datasets/speechCmd/data'
#out_path = '../../shared_datasets/speechCmd/balanced'

if not os.path.exists(out_path):
    os.makedirs(out_path)

classes = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
#classes = ['up', 'down', 'left', 'right']

items_per_class = 1000

f = os.listdir(in_path)
if '.DS_Store' in f:
    f.remove('.DS_Store')

lens = []
for i in f:
    if i in classes:
        print (i)
        p = os.path.join(in_path, i)

        s = os.listdir(p)
        if '.DS_Store' in s:
            s.remove('.DS_Store')
        lens.append(len(s))

min_len = min(lens)
max_len = max(lens)
mean_len = np.mean(lens)
tot = np.sum(lens)

print (min_len, mean_len, max_len, tot)

for label in f:
    if label in classes:
        label_path = os.path.join(in_path, label)
        sounds = os.listdir(label_path)
        if '.DS_Store' in sounds:
            sounds.remove('.DS_Store')
        random.shuffle(sounds)
        random.shuffle(sounds)
        for n in range(items_per_class):
            name = sounds[n].split('.')[0]
            curr_in_path = os.path.join(label_path, sounds[n])
            curr_out_path = os.path.join(out_path, name+'_'+label+'.wav')
            copy(curr_in_path, curr_out_path)
            print ('')
            print (curr_in_path)
            print (curr_out_path)
