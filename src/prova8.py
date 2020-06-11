import numpy as np
import os


f = '/Users/eric/Desktop/temp/culo'

e = [os.path.join(f, x) for x in os.listdir(f)]

for i in e:
    a = np.load(i, allow_pickle=True).item()
    culo = a[0]['train_acc']
    v = a[0]['val_acc']
    t = a[0]['test_acc']

    print (culo,v,t)




'''

nsynth
98.1, 69.94

goodsounds
100, 100
iemocap
99.88, 96.5

librispeech
97.6, 91.85,
