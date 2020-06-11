import numpy as np
import matplotlib.pyplot as plt
'''
Plot the loss trand through training epochs (as in the original paper)
'''

path = '../bin/results_iemocap_randsplit_spectrum_fast_exp208_run5.npy'

a = np.load(path, allow_pickle=True).item()

train_loss = a[0]['train_loss_hist']
val_loss = a[0]['val_loss_hist']
train_feature_loss = a[0]['train_feature_loss_hist']
val_feature_loss = a[0]['val_feature_loss_hist']

plt.figure(1)
plt.subplot(211)
plt.title('cross entropy loss')
plt.plot(train_loss, label='train loss',color='blue', linestyle='-')
plt.plot(val_loss, label='val loss',color='orange', linestyle='-')
plt.xticks(range(11),[1,2,3,4,5,6,7,8,9,10,11])
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
plt.subplot(212)
plt.title('anti transfer loss')
plt.plot(train_feature_loss, label='train loss',color='blue', linestyle='-')
plt.plot(val_feature_loss, label='val loss',color='orange', linestyle='-')
plt.xticks(range(11),[1,2,3,4,5,6,7,8,9,10,11])
plt.subplots_adjust(wspace=None, hspace=0.6)
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
