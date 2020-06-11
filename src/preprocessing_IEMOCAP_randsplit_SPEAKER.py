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
AUGMENTATION = eval(cfg.get('feature_extraction', 'augmentation'))
NUM_AUG_SAMPLES = eval(cfg.get('feature_extraction', 'num_aug_samples'))
SEGMENTATION = True
INPUT_IEMOCAP_FOLDER = cfg.get('preprocessing', 'input_iemocap_folder')
OUTPUT_FOLDER = '../dataset/matrices/iemocap_speaker'



if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if AUGMENTATION:
    print ('Augmentation: ' + str(AUGMENTATION) + ' | num_aug_samples: ' + str(NUM_AUG_SAMPLES) )
else:
    print ('Augmentation: ' + str(AUGMENTATION))

print ('Segmentation: ' + str(SEGMENTATION))
print ('Features type: ' + str(FEATURES_TYPE))

#put NONE in one key to not preprocess that label
#and change num_classes_IEMOCAP
num_classes_IEMOCAP = 4
num_speakers_IEMOCAP = 5


label_to_int = {'neu':0,
                'ang':1,
                'hap':2,
                'exc':None,
                'sad':3,
                'fru':None,
                'fea':None,
                'sur':None,
                'dis':None,
                'oth':None,
                'xxx':None}

wavname = 'Ses01F_impro01_F001.wav'
#wavname = 'Ses01M_script01_2_F003.wav'

def get_max_length_IEMOCAP(input_list):
    '''
    get longest audio file (insamples) for eventual zeropadding
    '''
    max_file_length, sr = uf.find_longest_audio_list(input_list)
    max_file_length = int(max_file_length * sr)

    return max_file_length

def get_label_IEMOCAP(wavname):
    '''
    compute one hot label starting from wav filename
    '''
    if "Ses01" in wavname:
        label = 0
    elif "Ses02" in wavname:
        label = 1
    elif "Ses03" in wavname:
        label = 2
    elif "Ses04" in wavname:
        label = 3
    elif "Ses05" in wavname:
        label = 4
    else:
        raise NameError("wrong IEMOCAP label name")

    label = uf.onehot(label, num_speakers_IEMOCAP)
    return label

def get_sounds_list(input_folder=INPUT_IEMOCAP_FOLDER):
    '''
    get list of all sound paths in the dataset
    '''
    paths = []
    contents = os.listdir(input_folder)
    contents = list(filter(lambda x: 'Session' in x, contents))
    #iterate sessions
    for session in contents:
        session_path = os.path.join(input_folder, session, 'sentences/wav')
        dialogs = os.listdir(session_path)
        #iterate dialogs
        for dialog in dialogs:
            dialog_path = os.path.join(session_path, dialog)
            utterances = os.listdir(dialog_path)
            #iterate utterance files
            for utterance in utterances:
                utterance_path = os.path.join(dialog_path, utterance)
                paths.append(utterance_path)

    return paths

def filter_labels(sounds_list):
    '''
    filter only sounds that are in the
    selected label_to_int dict
    '''
    filtered_list = []
    for sound in sounds_list:
        label = get_label_IEMOCAP(sound)
        if type(label) == np.ndarray:  #if not none
            filtered_list.append(sound)

    return filtered_list



def filter_actors_IEMOCAP(sounds_list, foldable_item):
    '''
    this function simply returns the input string as a list
    NOTE THAT DOING THIS WE DO NOT SPLIT DATASET ACCORDING TO
    DIFFERENT ACTORS!!!
    we only xfold and tr/val/test split in order to not have segments of the
    same recordings divided in different sets
    '''
    key = 'Ses0' + str(foldable_item + 1)
    curr_list = list(filter(lambda x: key in x, sounds_list))

    return curr_list

def main():
    '''
    custom preprocessing routine for the iemocap dataset
    '''
    print ('')
    print ('Setting up preprocessing...')
    print('')
    sounds_list = get_sounds_list(INPUT_IEMOCAP_FOLDER)  #get list of all soundfile paths

    #change this to have only 4 labels
    filtered_list = filter_labels(sounds_list)  #filter only sounds of certain labels

    #filter non-wav files
    filtered_list = list(filter(lambda x: x[-3:] == "wav", filtered_list))  #get only wav
    random.shuffle(filtered_list)

    #filtered_list = sounds_list

    num_foldables = 5

    if SEGMENTATION:
        max_file_length = 1
    else:
        max_file_length=get_max_length_IEMOCAP(filtered_list)  #get longest file in samples
    num_files = len(filtered_list)
    #init predictors and target dicts
    predictors = {}
    target = {}
    #create output paths for the npy matrices
    appendix = '_' + FEATURES_TYPE
    if AUGMENTATION:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, 'iemocap_randsplit' + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, 'iemocap_randsplit' + appendix + '_aug' + str(NUM_AUG_SAMPLES) + '_target.npy')
    else:
        predictors_save_path = os.path.join(OUTPUT_FOLDER, 'iemocap_randsplit' + appendix + '_predictors.npy')
        target_save_path = os.path.join(OUTPUT_FOLDER, 'iemocap_randsplit' + appendix + '_target.npy')
    index = 1  #index for progress bar

    for i in filtered_list:
        #print progress bar
        #fold_string = '\nPreprocessing foldable item: ' + str(index) + '/' + str(num_foldables)
        #print (fold_string)
        print ('\nPreprocessing files')
        #get foldable item DIVIDING BY ACTORS. Every session hae 2 actors
        curr_list = [i]

        #preprocess all sounds of the current actor
        #args:1. listof soundpaths of current actor, 2. max file length, 3. function to extract label from filepath
        curr_predictors, curr_target = pre.preprocess_foldable_item(curr_list, max_file_length, get_label_IEMOCAP)
        #print ('cazzo', curr_predictors.shape)
        #append preprocessed predictors and target to the dict
        if curr_predictors.shape[0] != 0:
            #print (curr_predictors.shape)
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
