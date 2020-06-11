import os
import numpy as np
import preprocessing_utils
import soundfile as sf
from scipy.signal import stft
import utility_functions as uf
import random
import sys
import pandas


preprocessing_ID = 'reduced'
max_items = 280000 / 5
input_path = '../dataset/nsynth'
#input_path = '/Users/eric/Desktop/nsynth'
output_path = '../dataset/matrices/nsynth'
#output_path = '/Users/eric/Desktop/cazzo'

target_feature = 'instrument_family'

# Run script
if __name__ == '__main__':
    # splits
    splits = os.listdir(input_path)
    splits = list(filter(lambda x: '.DS_Store' not in x, splits))

    splits = ['nsynth-test','nsynth-valid','nsynth-train']

    splits = [os.path.join(input_path, x) for x in splits]
    #iterate train, validation, test datasets
    for split in splits:
        print ('preprocessing: ' + str(split))
        predictors = []
        target = []
        audio_folder = os.path.join(split, 'audio')
        table_path = os.path.join(split, 'examples.json')
        table = pandas.read_json(table_path)
        sounds = os.listdir(audio_folder)
        sounds = [os.path.join(audio_folder, x) for x in sounds]
        random.shuffle(sounds)
        n_sounds = len(sounds)
        print ('num sounds: ', n_sounds)
        #iterate all sounds of a split
        index = 0
        for sound_path in sounds:
            #compute predictors
            if index <= max_items:
                samples, sr = sf.read(sound_path)
                fft = preprocessing_utils.spectrum_fast(samples)
                #compute target
                sound_name = sound_path.split('/')[-1].split('.')[0]
                label = table[sound_name][target_feature]

                predictors.append(fft)
                target.append(label)

                uf.print_bar(index, min(n_sounds, max_items))
                index += 1

        if 'nsynth-valid' in split:
            name_tag = 'validation'
        elif 'nsynth-train' in split:
            name_tag = 'training'
        elif 'nsynth-test' in split:
            name_tag = 'test'

        #save matrix of current split
        if not os.path.exists(os.path.join(output_path, 'nsynth')):
            os.makedirs(os.path.join(output_path, 'nsynth'))
        predictors_path = os.path.join(output_path, 'nsynth', preprocessing_ID + '_' + name_tag + '_predictors.npy')
        target_path = os.path.join(output_path, 'nsynth', preprocessing_ID + '_' +  name_tag + '_target.npy')
        print (split, np.array(predictors).shape)
        np.save(predictors_path, np.array(predictors))
        np.save(target_path, np.array(target))
