# Anti-Transfer Learning for Task Invariance in Convolutional Neural Networks for Speech Processing
This repository supports the above [paper](https://arxiv.org/abs/2006.06494) submitted to the Neural Networks journal.
This very simple code permits to apply AT loss to any PyTorch convolutional neural network design.

## Installation
To install all dependencies run:
```bash
pip install -r requirements.txt
```

## Concept
While transfer learning assumes that the learning process for a target task will benefit from re-using representations learned for another task, **anti-transfer**  avoids the learning of representations that have been learned for  an *orthogonal task*,  i.e., one that is not relevant and potentially confounding for the  target task, such as speaker identity for speech recognition or speech content for emotion recognition. This extends the potential use of pre-trained models that have become increasingly available. In anti-transfer learning, we penalize similarity between activations of a net-work being trained on a target task and another one previously trained on an orthogonal task, which yields more suitable representations. This leads to better generalization and provides a degree of control over correlations that are spurious or undesirable, e.g. to avoid social bias.

## Usage
The **example_usage.py** script provides a working example of a training performed with anti-transfer. The usage is quite straightforward: you should first train a model on an orthogonal. Then train another identical model applying anti-transfer using the the pre-trained one as feature extractor.
