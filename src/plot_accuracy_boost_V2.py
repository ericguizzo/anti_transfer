import random
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt
'''
Plot the overall average test accuracy improvement in various test cases
(as in the original paper)
'''

overall_train = ((75.5-69) + (74.5-67.8) + (93.9-91.8) + (36.4-42.2) + (99.84-98.45) + (99.49-97.94) + (98.89-97.23)) / 6
print (overall_train)

#test
sound_goodness =((86.3-83.8) + (34.3-22.8) ) / 2
ser = ((66.5-63.7) + (61.3-57.2) ) / 2
asr = ((96.6-95.32)+(95.2-93.67)+(91.38-90.44)) / 3
overall = ((86.3-83.8) + (34.3-22.8) + (66.5-63.7) + (61.3-57.2) + (96.6-95.32)+(95.2-93.67)+(91.38-90.44)) / 6
bigger_pre = ((86.3-83.8) + (30-22.8) + (66.9-63.7) + (61.1-57.2)) / 4
same_pre = ((66.5-63.7) + (61.3-57.1) + (85.7-83.8) + (34.3-22.8))/4
single_at = ((95.7-95.32)+(95.57-95.32)+(94.81-93.67)+(94.91-93.67)+(91.38-90.44)+(90.99-91.38)) / 6
double_at = ((96.60-95.32)+(95.64-95.32)+(94.91-93.67)+(95.2-93.67)+ (90.98-90.44) + (90.67-90.44)) / 6

asr = np.round(asr, decimals=2)
sound_goodness = np.round(sound_goodness, decimals=2)
ser = np.round(ser, decimals=2)
overall = np.round(overall, decimals=2)
bigger_pre = np.round(bigger_pre, decimals=2)
same_pre = np.round(same_pre, decimals=2)
single_at = np.round(single_at, decimals=2)
double_at = np.round(double_at, decimals=2)

plt.rcdefaults()
fig, ax = plt.subplots()

modalities = ('Overall', 'Automatic \nSpeech \nRecognition', 'Speech \nEmotion \nRecognition', 'Sound \nGoodness \nEstimation', 'Pre-train on \nbigger dataset \n(GS + IEMOCAP)', 'Pre-train on \nsame dataset \n(GS + IEMOCAP)', 'Single AT \n(GSC)', 'Dual AT \n(GSC)')
y_pos = np.arange(len(modalities))

performance = [overall, asr, ser, sound_goodness, bigger_pre, same_pre, single_at, double_at]

ax.barh(y_pos, performance, align='center', color='orange', alpha=0.5)

for i, v in enumerate(performance):
    ax.text(v , i +0.1, str(v)+'', color='black', fontweight='bold')

ax.set_yticks(np.arange(len(modalities)))
ax.set_yticklabels(modalities)
ax.set_xticks(np.arange(9))
ax.set_xticklabels(['','1','2','3','4','5','6','7', ''])

ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Test Accuracy Boost (percentage points)')
ax.set_title('Test Accuracy Boost In Different Modalities')


plt.show()
pyplot.show()
