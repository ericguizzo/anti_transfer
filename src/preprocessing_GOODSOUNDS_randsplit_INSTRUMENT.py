import numpy as np
import pandas as pd
import sqlite3
import sys
import utility_functions as uf
import preprocessing_utils as pre
import random
import numpy as np
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

#get values from config file
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SR = cfg.getint('sampling', 'sr_target')
AUGMENTATION = eval(cfg.get('feature_extraction', 'augmentation'))
NUM_AUG_SAMPLES = eval(cfg.get('feature_extraction', 'num_aug_samples'))
SEGMENTATION = eval(cfg.get('feature_extraction', 'segmentation'))
INPUT_GOODSOUNDS_FOLDER = cfg.get('preprocessing', 'input_goodsounds_folder')
OUTPUT_FOLDER = '../dataset/matrices/goodsounds_instrument'
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if AUGMENTATION:
    print ('Augmentation: ' + str(AUGMENTATION) + ' | num_aug_samples: ' + str(NUM_AUG_SAMPLES) )
else:
    print ('Augmentation: ' + str(AUGMENTATION))

print ('Segmentation: ' + str(SEGMENTATION))
print ('Features type: ' + str(FEATURES_TYPE))


#read sqlite table
database_path = os.path.join(INPUT_GOODSOUNDS_FOLDER, 'database.sqlite')
#a = '/Users/eric/Desktop/temp/database.sqlite'
cnx = sqlite3.connect(database_path)
#cnx = sqlite3.connect(a)

df = pd.read_sql_query("SELECT * FROM sounds", cnx)
packs_df = pd.read_sql_query("SELECT * FROM packs", cnx)
'''
for i in range(df.shape[0]):
    print (df.iloc[i])
    print ('')
sys.exit(0)
takes_df = pd.read_sql_query("SELECT * FROM takes", cnx)
for i in range(takes_df.shape[0]):
    print (takes_df.iloc[i])
    print ('')
'''


#create dict with ID: pack_name couples
packs = {}
n_rows_packs = packs_df.shape[0]
for i in range(n_rows_packs):
    id = packs_df.iloc[i]['id']
    name = packs_df.iloc[i]['name']
    packs[id] = name

#build good/bad dict
n_rows = df.shape[0]
instruments = []
good_sounds = []
bad_attack = []
bad_dynamics = []
bad_timbre = []
bad_pitch = []
table = {}

for i in range(n_rows):
    try:
        lable_final = 0
        lable = str(df.iloc[i]['klass'])
        filename = str(df.iloc[i]['pack_filename'])
        pack_id = int(df.iloc[i]['pack_id'])
        pack_name = str(packs[pack_id])
        instrument = str(df.iloc[i]['instrument'])
        tuple_id = (pack_name, filename)

        #take only single sounds
        if 'scale' not in lable:
            #bad-pitch are just a few, so take only the double-lableled ones
            if 'bad-pitch' in lable:
                bad_pitch.append(tuple_id)
                lable_final = 'bad-pitch'
                table[tuple_id] = {'instrument':instrument, 'lable':lable_final}
                if instrument not in instruments:
                    instruments.append(instrument)
            #take only single lables:
            if ' ' not in lable and lable.count('bad') < 2:
                if 'good-sound' in lable:
                    good_sounds.append(tuple_id)
                    lable_final = 'good-sound'
                if 'bad-dynamics' in lable:
                    bad_dynamics.append(tuple_id)
                    lable_final = 'bad-dynamics'
                if 'bad-attack' in lable:
                    bad_attack.append(tuple_id)
                    lable_final = 'bad-attack'
                if 'bad-timbre' in lable or 'bad-richness'in lable:
                    bad_timbre.append(tuple_id)
                    lable_final = 'bad-timbre'

                if lable_final != 0:
                    #print (tuple_id)
                    table[tuple_id] = {'instrument':instrument, 'lable':lable_final}
                    if instrument not in instruments:
                        instruments.append(instrument)
    except ValueError as e:
        print (e)

#shuffle and balance data
min_occur = min(len(good_sounds), len(bad_timbre), len(bad_attack), len(bad_dynamics), len(bad_pitch))
random.shuffle(good_sounds)
random.shuffle(bad_pitch)
random.shuffle(bad_attack)
random.shuffle(bad_dynamics)
random.shuffle(bad_timbre)

#print (len(table))
#print ((len(good_sounds) + len(bad_timbre) + len(bad_attack) + len(bad_dynamics) + len(bad_pitch)))

good_sounds = good_sounds[:min_occur-1]
bad_pitch = bad_pitch[:min_occur-1]
bad_attack = bad_attack[:min_occur-1]
bad_dynamics = bad_dynamics[:min_occur-1]
bad_timbre = bad_timbre[:min_occur-1]
#print ((len(good_sounds) , len(bad_timbre) , len(bad_attack) , len(bad_dynamics) , len(bad_pitch)))
merged_list = good_sounds + bad_dynamics + bad_pitch + bad_timbre + bad_attack
for i in list(table.keys()):
    if i not in merged_list:
        del table[i]
#print (len(table))
#print ((len(good_sounds) + len(bad_timbre) + len(bad_attack) + len(bad_dynamics) + len(bad_pitch)))


#!!!!!!!!!!!now you have table{(folder, filename): string_lable}

num_classes_GOODSOUNDS = 5
num_instruments = 12
num_foldables = 12 #number of instruments
SEQUENCE_LENGTH = 6

lable_to_int = {'good-sound':0,
                'bad-dynamics':1,
                'bad-pitch':2,
                'bad-timbre':3,
                'bad-attack':4}


instrument_to_int = {'flute': 0,
                     'clarinet': 1,
                     'trumpet':2,
                     'violin':3,
                     'cello':4,
                     'sax_alto':5,
                     'sax_tenor':6,
                     'sax_baritone':7,
                     'sax_soprano':8,
                     'oboe':9,
                     'piccolo':10,
                     'bass':11}


def get_max_length(input_list):
    '''
    get longest audio file (insamples) for eventual zeropadding
    '''
    max_file_length, sr = uf.find_longest_audio_list2(input_list)
    max_file_length = int(max_file_length * sr)

    return max_file_length

def get_lable_instrument_GOODSOUNDS(wavname):
    '''
    compute one hot lable starting from wav filename
    '''
    unpacked = wavname.split('/')
    pack = unpacked[-3]
    name = unpacked[-1]
    tuple_id = (pack, name)
    lable = table[tuple_id]['instrument']
    int_lable = instrument_to_int[lable]
    output = uf.onehot(int_lable, num_instruments)
    print (pack, lable, int_lable)

    return int_lable

def get_lable_GOODSOUNDS(wavname):
    '''
    compute one hot lable starting from wav filename
    '''
    unpacked = wavname.split('/')
    pack = unpacked[-3]
    name = unpacked[-1]
    tuple_id = (pack, name)
    lable = table[tuple_id]['lable']
    int_lable = lable_to_int[lable]
    output = uf.onehot(int_lable, num_classes_GOODSOUNDS)

    return output

def get_sounds_list(input_folder=INPUT_GOODSOUNDS_FOLDER):
    '''
    get list of all sound paths in the dataset
    '''
    paths = []
    contents = [os.path.join(input_folder, x) for x in os.listdir(input_folder)]
    #iterate "packs" folders
    for f in contents:
        w = ['_', '#']
        mics = os.listdir(f)
        #only one microphone take
        mics = list(filter(lambda x:'_' not in x and '#' not in x and '.csv' not in x, mics))
        sf = os.path.join(f, mics[0])
        sounds = [os.path.join(sf, x) for x in os.listdir(sf)]
        sounds = list(filter(lambda x: '.wav' in x, sounds))
        for s in sounds:
            unpacked = s.split('/')
            pack = unpacked[-3]
            name = unpacked[-1]
            tuple_id = (pack, name)
            if tuple_id in table.keys():  #discard if not selected in table dict
                paths.append(s)

    return paths


def filter_instruments_GOODSOUNDS(sounds_list, foldable_item):
    target_instrument = instruments[foldable_item]
    #curr_list = list(filter(lambda x: curr_instrument in x, sounds_list))
    curr_list = []
    for s in sounds_list:
        unpacked = s.split('/')
        pack = unpacked[-3]
        name = unpacked[-1]
        tuple_id = (pack, name)
        curr_instrument = table[tuple_id]['instrument']
        #print (target_instrument, curr_instrument)
        if curr_instrument == target_instrument:
            curr_list.append(s)

    return curr_list

def main():
    '''
    custom preprocessing routine for the iemocap dataset
    '''
    print ('')
    print ('Setting up preprocessing...')
    print('')
    sounds_folder = os.path.join(INPUT_GOODSOUNDS_FOLDER, 'sound_files')
    sounds_list = get_sounds_list(sounds_folder)  #get list of all soundfile paths
    random.shuffle(sounds_list)
    #max_file_length=get_max_length(sounds_list)  #get longest file in samples
    max_file_length = SR * SEQUENCE_LENGTH #pre-computed
    num_files = len(sounds_list)
    #init predictors and target dicts

    predictors = {}
    target = {}
    #create output paths for the npy matrices
    appendix = '_' + FEATURES_TYPE
    if AUGMENTATION:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, 'goodsounds_INSTRUMENT_randsplit' + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, 'goodsounds_INSTRUMENT_randsplit' + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_target.npy')
    else:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, 'goodsounds_INSTRUMENT_randsplit' + appendix + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, 'goodsounds_INSTRUMENT_randsplit' + appendix + '_target.npy')
    index = 1  #index for progress bar

    #sounds_list = sounds_list[:10]
    print ('\nPreprocessing files')
    for i in sounds_list:
        #print progress bar
        #get foldable item DIVIDING BY ACTORS. Every session hae 2 actors
        curr_list = [i]
        #get foldable item DIVIDING BY ACTORS. Every session hae 2 actors
        #print (instruments[i],len(curr_list))

        curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, max_file_length, get_lable_instrument_GOODSOUNDS)
        #print (curr_predictors.shape)
        #append preprocessed predictors and target to the dict
        if curr_predictors.shape[0] != 0:
            print (curr_predictors.shape)
            predictors[i] = curr_predictors
            target[i] = curr_target

        uf.print_bar(index, num_files)

        index +=1

    #save dicts

    print ('\nSaving matrices...')
    np.save(predictors_save_path, predictors)
    np.save(target_save_path, target)
    #print dimensions
    count = 0
    predictors_dims = 0
    keys = list(predictors.keys())
    for i in keys:
        count += predictors[i].shape[0]
    pred_shape = np.array(predictors[keys[0]]).shape[1:]
    tg_shape = np.array(target[keys[0]]).shape[1:]
    print ('')
    print ('MATRICES SUCCESFULLY COMPUTED')
    print ('')
    print ('Total number of datapoints: ' + str(count))
    print (' Predictors shape: ' + str(pred_shape))
    print (' Target shape: ' + str(tg_shape))



if __name__ == '__main__':
    main()





#print (table)
