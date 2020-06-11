This repository supports the paper "Blissful Ignorance: Anti-Transfer Learning for Task Invariance" submitted for NeurIPS2020.
Running this code will produce the results mentioned in the paper, using the same configuration as described in the paper.

Code is based on Python 3.5


## INSTALL DEPENDENCIES
```bash
pip3 install numpy scipy librosa configparser matplotlib  torch torchvision essentia pandas soundfile sklearn xlswriter
```

## PREPARE DATA

### Iemocap
* Download dataset: https://sail.usc.edu/iemocap/
* Open the config/config.ini file and put the dataset path on [preprocessing]-input_iemocap_folder
* Run the following scripts to pre-process the dataset both with random and speaker-wise train/val/test split for emotion recognition and with random split for speaker recognition.
```bash
python3 preprocessing_IEMOCAP_randsplit.py
python3 preprocessing_IEMOCAP_actorsplit.py
python3 preprocessing_IEMOCAP_randsplit_SPEAKER.py
```
### Good-Sounds
* Download dataset: https://www.upf.edu/web/mtg/good-sounds
* Open the config/config.ini file and put the dataset path on [preprocessing]-input_good-sounds_folder
* Run the following scripts to pre-process the dataset both with random and instrument-wise train/val/test split for sound goodness estimation and with random split for instrument recognition.
```bash
python3 preprocessing_GOODSOUNDS_randsplit.py
python3 preprocessing_GOODSOUNDS_actorsplit.py
python3 preprocessing_GOODSOUNDS_randsplit_INSTRUMENT.py
```
### Librispeech
* Follow these instructions to download and prepare the dataset: https://github.com/bepierre/SpeechVGG. (We used the 100 hours version)
* Open the config/config.ini file and put the dataset path on [preprocessing]-input_librispeech_folder
* Run the following script to pre-process the dataset.
```bash
python3 preprocessing_LIBRISPEECH.py
```
### Nsynth
* Download dataset: https://magenta.tensorflow.org/datasets/nsynth
* Put training, validation and test folders in the same folder
* Open the config/config.ini file and put the dataset path on [preprocessing]-input_nsynth_folder
* Run the following script to pre-process the dataset.
```bash
python3 preprocessing_NSYNTH.py
```

##PRE-TRAIN FEATURE DEEP EXTRACTORS
Separately run the following scripts to pre-train the VGG networks used to compute the deep feature losses:
```bash
python3 pretrain_vgg_iemocap.py
python3 pretrain_vgg_goodsounds.py
python3 pretrain_vgg_librispeech.py
python3 pretrain_vgg_nsynth.py
```
Depending on the GPU, it may be necessary to modify the batch_size variable in these scripts.

## GENERATE PLOTS
Run the following bash scripts to produce the plots included in the paper.
Results values are hard-coded in the scripts.
```bash
python3 plot_accuracy_boost.py
python3 plot_loss_epochs.py
python3 plot_perLayer_improvement.py
```
