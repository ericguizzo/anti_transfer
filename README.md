This repository supports the paper "Blissful Ignorance: Anti-Transfer Learning for Task Invariance" submitted for NeurIPS2020.
Running this code will produce the results mentioned in the paper. Code is based on Python 3.5


## INSTALL DEPENDENCIES
```bash
pip3 install numpy scipy librosa configparser matplotlib  torch torchvision essentia pandas soundfile sklearn xlswriter
```

## PREPARE DATA

### Iemocap
* Download dataset: https://sail.usc.edu/iemocap/
* Open the config/config.ini file and put the dataset path on [preprocessing]-input_iemocap_folder
* Run the following to pre-process the dataset both with random and speaker-wise train/val/test split for emotion recognition and with random split for speaker recognition
```bash
python3 preprocessing_IEMOCAP_randsplit.py
python3 preprocessing_IEMOCAP_actorsplit.py
python3 preprocessing_IEMOCAP_randsplit__SPEAKER.py

```

## GENERATE PLOTS
Run the following bash scripts to produce the plots included in the paper.
Results values are hard-coded in the scripts.
```bash
python3 plot_accuracy_boost.py
python3 plot_loss_epochs.py
python3 plot_perLayer_improvement.py
```
