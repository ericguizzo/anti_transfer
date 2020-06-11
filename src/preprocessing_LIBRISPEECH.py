import os
import numpy as np
from argparse import ArgumentParser
import preprocessing_utils
import soundfile as sf
from scipy.signal import stft
import h5py
import configparser
import loadconfig
#functions taken from https://github.com/bepierre/SpeechVGG

config = loadconfig.load()
cfg = configparser.ConfigParser()
cfg.read(config)

#get values from config file
input_folder = cfg.get('preprocessing', 'input_librispeech_folder')


preprocessing_ID = '4sec_inv'

def parse_args():
    parser = ArgumentParser(description='Preprocessing of soundfiles')

    parser.add_argument(
        '-data', '--data',
        type=str, default=input_folder,
        help='dataset to load'
    )

    parser.add_argument(
        '-dest_path', '--dest_path',
        type=str, default='../dataset/matrices/librispeech',
        help='destination of processed data'
    )

    parser.add_argument(
        '-classes', '--classes',
        type=int, default=1000,
        help='number of classes (in our case words)'
    )
    return parser.parse_args()

# Run script
if __name__ == '__main__':
    # Parse command-line arguments
    args = parse_args()

    # splits
    split_dir = args.data+'/split/'
    splits = [ name for name in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, name)) ]
    dict_file = args.data + '/word_labels/{}-mostfreq'.format(args.classes)
    dictionary = open(dict_file, "r").read().split('\n')
    if dictionary[-1] == '': del dictionary[-1]

    max_len = 0 #max stft len to be counted

    #find max len
    durs = []
    for split in splits:
        word_file = args.data + '/word_labels/' + split + '-selected-' + str(args.classes) + '.txt'
        with open(word_file) as wf:
            for line in wf.readlines():
                # remove endline if present
                line = line[:line.find('\n')]
                segment_name, _, time_beg, time_len, word, _ = line.split(' ')
                start = int(float(time_beg) * 16000)
                end = int((float(time_beg) + float(time_len)) * 16000)
                dur = end - start
                durs.append(dur)
    max_dur = max(durs)

    #FORCE 4-SECONDS zeropadding
    #max_dur = 16000 * 4

    del durs
    for split in splits:
        #init datasets
        predictors = []
        target = []

        # Create output directories if don't exist
        os.makedirs(args.dest_path + '/' + split, exist_ok=True)

        print('Pre-processing: {}'.format(split))

        # Get file names
        word_file = args.data + '/word_labels/' + split + '-selected-' + str(args.classes) + '.txt'

        current_file_name = ''
        audio = 0

        with open(word_file) as wf:

            segment_num = 0

            for line in wf.readlines():

                # remove endline if present
                line = line[:line.find('\n')]
                segment_name, _, time_beg, time_len, word, _ = line.split(' ')

                file_name = args.data + '/split/' + split + '/' + segment_name.replace('-', '/')[:segment_name.rfind('-')+1] + segment_name + '.flac'
                if file_name != current_file_name:
                    audio = sf.read(file_name)
                    audio = audio[0]
                    current_file_name = file_name
                    segment_num = 0

                start = int(float(time_beg) * 16000)
                end = int((float(time_beg) + float(time_len)) * 16000)
                cut = audio[start:end]
                padded = np.zeros(max_dur)
                padded[:len(cut)] = cut
                '''
                f, t, seg_stft = stft(padded,
                                      window='hamming',
                                      nperseg=256,
                                      noverlap=128)
                predictors.append(np.abs(seg_stft))
                '''
                seg_fft = preprocessing_utils.spectrum_fast(padded)
                predictors.append(seg_fft)
                target.append(float(dictionary.index(word.lower())))

                segment_num = segment_num + 1


            if 'dev' in split:
                name_tag = 'validation'
            elif 'train' in split:
                name_tag = 'training'
            elif 'test' in split:
                name_tag = 'test'

            predictors_path = os.path.join(args.dest_path, 'librispeech', preprocessing_ID + '_' + name_tag + '_predictors.npy')
            target_path = os.path.join(args.dest_path, 'librispeech', preprocessing_ID + '_' +  name_tag + '_target.npy')
            print (split, np.array(predictors).shape)
            np.save(predictors_path, np.array(predictors))
            np.save(target_path, np.array(target))
