# Anti-Transfer Learning for Task Invariance in Convolutional Neural Networks for Speech Processing
This repository supports the above paper submitted to Neural Networks journal.
This code produces the results mentioned in the paper, using the same configuration described in the original work.

## REQUIREMENTS
We used these exact packages versions:
* python 3.5
* ipython 7.1.1
* numpy 1.15.4
* scipy 1.4.1
* librosa 0.6.3
* matplotlib 3.0..3
* torch 1.4.0
* torchvision 0.2.2.post3
* essentia 2.1b5.dev532
* pandas 0.24.2
* soundfile 0.9.0.post1
* scikit-learn 0.20.3
* xlswriter 1.1.8
* configparser 2.19

To install everything at once please run:
```bash
pip install -r requirements.txt
```
## PREPARE DATA
Please follow these instructions in the correct sequence to prepare all datasets for the trainings. Since we evaluated our approach on many different datasets, this stage may take a while.

## Noisy Speech
* Clone the official MS-SNSD repository: https://github.com/microsoft/MS-SNSD
* Open the config/config.ini file and put the repo path on [preprocessing]-input_noisyspeech_repo and the path in which you want to generate the dataset (since it is scalable) at [preprocessing]-generated_noisyspeech_folder.
* Run the following scripts to generate and preprocess the dataset:
```bash
python3 noisyspeech_synthesizer.py
python3 preprocessing_NOISYSPEECH_randsplit.py
```

## Google Speech Commands
* Download the dataset at this link: http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz
* Open the config/config.ini file and put the dataset path on [preprocessing]-input_gsc_raw_folder and the path in which you want to generate the augmented data on [preprocessing]-augmented_gsc_folder
* Run the following scripts to balance and augment the dataset with the same background noises of Noisy Speech:
```bash
python3 balance_gsc.py
python3 noisyspeech_synthesizer_gsc.py
python3 noisyspeech_synthesizer_gsc_MORE.py
```
* Run the following scripts to preprocess the plain and the augmented versions of the dataset:
```bash
python3 preprocessing_GSC_randsplit.py
python3 preprocessing_GSC_NOISY_randsplit.py
python3 preprocessing_GSC_NOISY_MORE_randsplit.py
```
### Iemocap
* Follow these instructions to download the dataset: https://sail.usc.edu/iemocap/
* Open the config/config.ini file and put the dataset path on [preprocessing]-input_iemocap_folder
* In the config/config.ini file put [pretraining]-sequence_length = 4
* Run the following scripts to pre-process the dataset both with random and speaker-wise train/val/test split for emotion recognition and with random split for speaker recognition:
```bash
python3 preprocessing_IEMOCAP_randsplit.py
python3 preprocessing_IEMOCAP_actorsplit.py
python3 preprocessing_IEMOCAP_randsplit_SPEAKER.py
```
### Good-Sounds
* Download dataset: https://www.upf.edu/web/mtg/good-sounds
* Open the config/config.ini file and put the dataset path on [preprocessing]-input_good-sounds_folder
* In the config/config.ini file put [pretraining]-sequence_length = 6
* Run the following scripts to pre-process the dataset both with random and instrument-wise train/val/test split for sound goodness estimation and with random split for instrument recognition:
```bash
python3 preprocessing_GOODSOUNDS_randsplit.py
python3 preprocessing_GOODSOUNDS_actorsplit.py
python3 preprocessing_GOODSOUNDS_randsplit_INSTRUMENT.py
```
### Librispeech
* Follow these instructions to download and prepare the dataset: https://github.com/bepierre/SpeechVGG. (We used the 100 hours version)
* Open the config/config.ini file and put the dataset path on [preprocessing]-input_librispeech_folder
* In the config/config.ini file put [pretraining]-sequence_length = 4
* Run the following script to pre-process the dataset:
```bash
python3 preprocessing_LIBRISPEECH.py
```
### Nsynth
* Download dataset: https://magenta.tensorflow.org/datasets/nsynth
* Put training, validation and test folders in the same folder
* Open the config/config.ini file and put the dataset path on [preprocessing]-input_nsynth_folder
* In the config/config.ini file put [pretraining]-sequence_length = 6
* Run the following script to pre-process the dataset:
```bash
python3 preprocessing_NSYNTH.py
```

## PRE-TRAIN THE DEEP FEATURE EXTRACTORS
When the preprocessing stage is finished, separately run the following scripts to pre-train the feature extractors:
```bash
python3 pretrain_vgg_iemocap.py
python3 pretrain_vgg_goodsounds.py
python3 pretrain_vgg_librispeech.py
python3 pretrain_vgg_nsynth.py
```
The pre-trained convolution part of these VGG networks will be used to compute the deep feature losses in the actual trainings.
Depending on the used GPU, it may be necessary to modify the batch_size variable inside these scripts.

## TRAIN THE MODELS
When all pre-trainings are completed, run the actual trainings to obtain the results exposed in the original paper. The experiments_antitransfer folder contains all configuration files for the experiments to run. Each configuration file contains several single training instances. The file naming describes each configuration: ID_training-dataset_split-type_pretraining-dataset
* ID 1-7 are the baseline. Here no anti-transfer nor weight initialization is applied. We compute 20 times all baseline experiments to consider possible random fluctuations.
* In ID 10 weight initialization, but no anti-transfer is applied. We compute 3 times each configuration to consider random fluctuations.
* In ID 2039 we apply anti-transfer learning. Each configuration is tested computing the anti-transfer loss in each of the 13 convolution layers of the VGG16 networks.

To run these experiments use the experiments_manager.py script inside the ipython environment, applying the following positional parameters:
1. IDs of experiments to run (python list)
2. GPU ID (int)
3. first instance to compute (OPTIONAL, int)
4. last instance to compute (OPTIONAL, int)

example:
```python
run experiments_manager [20,21,22,23] 0
```

The results are automatically saved in the ../results/experiment_*ID directory. These items are saved for every experiment:
* description.txt file containing a brief description of the experiment configuration
* models folder containing all saved weights of the models
* results folder containing the individial results of each experiment instance (saved as numpy dictionaries), a list of all used parameters for each instance (in the results/parameters folder) and an immediately-readable .xls spreadsheet that shows the most important result values of each experiment instance.

Example spreadsheet (Emotion recognition on speaker-wise split IEMOCAP with pre-training on speaker recognition (IEMOCAP)):
<p align="center">
<img src="bin/example_xls1.png" width="400">
</p>

In the two "comment" columns the main parameters of an experiment instance are shown. Best loss/accuracy are highlighted in green for every column.


## GENERATE THE PLOTS
Run the following scripts to produce the plots included in the paper
(our result values are hard-coded in the scripts).
```bash
python3 plot_accuracy_boost.py
python3 plot_loss_epochs.py
python3 plot_perLayer_improvement.py
```
