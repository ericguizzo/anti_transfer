import utility_functions as uf
import preprocessing_utils as pre
import random
import numpy as np
import os, sys
import configparser
import loadconfig

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
FEATURES_TYPE = cfg.get('feature_extraction', 'features_type')
SR = cfg.getint('sampling', 'sr_target')
AUGMENTATION = False
NUM_AUG_SAMPLES = eval(cfg.get('feature_extraction', 'num_aug_samples'))
SEGMENTATION = True
INPUT_NOISYSPEECH_FOLDER = cfg.get('preprocessing', 'generated_noisyspeech_folder')
OUTPUT_FOLDER = cfg.get('preprocessing', 'output_folder')

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if AUGMENTATION:
    print ('Augmentation: ' + str(AUGMENTATION) + ' | num_aug_samples: ' + str(NUM_AUG_SAMPLES) )
else:
    print ('Augmentation: ' + str(AUGMENTATION))

print ('Segmentation: ' + str(SEGMENTATION))
print ('Features type: ' + str(FEATURES_TYPE))


print ('loading data...')
num_classes_noisyspeech = 11
noise_classes = ['AirConditioner', 'AirportAnnouncements', 'Babble', 'CopyMachine', 'Munching',
                'NeighborSpeaking', 'ShuttingDoor', 'SqueakyChair', 'Typing', 'VacuumCleaner',
                'WasherDryer']

num_classes_noisyspeech = len(noise_classes)
assoc_dict = {}
ind = 0
for i in noise_classes:
    assoc_dict[i] = ind
    ind += 1

sounds_list = os.listdir(INPUT_NOISYSPEECH_FOLDER)
if '.DS_Store' in sounds_list:
    sounds_list.remove('.DS_Store')
sounds_list = [os.path.join(INPUT_NOISYSPEECH_FOLDER, i) for i in sounds_list]

num_sounds = len(sounds_list)


def get_label_NOISYSPEECH(wavname):
    '''
    compute one hot label starting from wav filename
    '''
    #label = wavname.split('/')[-2]  #string_label
    label = wavname.split('/')[-1].split('.')[-2].split('_')[-1]
    int_label = assoc_dict[label]
    #print ('\ncazzo',wavname, label)
    one_hot_label = (uf.onehot(int(int_label), num_classes_noisyspeech))
    print (label, int_label)

    return one_hot_label


def main():
    '''
    custom preprocessing routine for the iemocap dataset
    '''
    print ('')
    print ('Setting up preprocessing...')
    print('')
    '''
    if SEGMENTATION:
        max_file_length = 1
    else:
        max_file_length=get_max_length_GSC(INPUT_GSC_FOLDER)  #get longest file in samples
    '''
    max_file_length = 16000
    #init predictors and target dicts
    predictors = {}
    target = {}
    #create output paths for the npy matrices
    appendix = '_' + FEATURES_TYPE

    if AUGMENTATION:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, 'noisyspeech_randsplit' + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, 'noisyspeech_randsplit' + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_target.npy')
    else:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, 'noisyspeech_randsplit' + appendix + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, 'noisyspeech_randsplit' + appendix + '_target.npy')

    index = 1  #index for progress bar
    print ('Preprocessing dataset:')
    random.shuffle(sounds_list)
    for i in sounds_list:
        #print progress bar
        #uf.print_bar(index, num_actors)
        #get foldable item
        curr_list = [i]
        #curr_list = [os.path.join(INPUT_GSC_FOLDER, x) for x in curr_list]
        #fold_string = '\nPreprocessing dataset: ' + str(index) + '/' + str(num_actors)
        #print (fold_string)
        #preprocess all sounds of the current actor
        #args:1. listof soundpaths of current actor, 2. max file length, 3. function to extract label from filepath
        #print (curr_list)
        try:
            curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, max_file_length, get_label_NOISYSPEECH, False)
            #append preprocessed predictors and target to the dict
            print (curr_predictors.shape)
            print (curr_target.shape)
            predictors[i] = curr_predictors
            target[i] = curr_target

        except Exception as e:
            print ('')
            print (e)  #PROBABLY SOME FILES ARE CORRUPTED


        uf.print_bar(index, num_sounds)

        index +=1
    #save dicts
    print ('\nSaving matrices')
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
