import random
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

overall_train = ((75.5-69) + (74.5-67.8) + (93.9-91.8) + (36.4-42.2)) / 4

#test
sound_goodness =((86.3-83.8) + (34.3-22.8) ) / 2
ser = ((66.5-63.7) + (61.3-57.2) ) / 2
overall = ((86.3-83.8) + (34.3-22.8) + (66.5-63.7) + (61.3-57.2)) / 4
bigger_pre = ((86.3-83.8) + (30-22.8) + (66.9-63.7) + (61.1-57.2)) / 4
same_pre = ((66.5-63.7) + (61.3-57.1) + (85.7-83.8) + (34.3-22.8))/4

sound_goodness = 7.0
ser = 3.45
overall = 5.25
bigger_pre = 4.2
same_pre = 5.1

plt.rcdefaults()
fig, ax = plt.subplots()

modalities = ('Overall', 'Speech \nEmotion \nRecognition', 'Sound \nGoodness \nEstimation', 'Pre-train on \nbigger dataset', 'Pre-train on \nsame dataset')
y_pos = np.arange(len(modalities))

performance = [overall, ser, sound_goodness, bigger_pre, same_pre]

ax.barh(y_pos, performance, align='center', color='orange', alpha=0.5)

for i, v in enumerate(performance):
    ax.text(v , i +0.1, str(v)+'%', color='black', fontweight='bold')

ax.set_yticks(np.arange(len(modalities)))
ax.set_yticklabels(modalities)
ax.set_xticks(np.arange(9))
ax.set_xticklabels(['','1%','2%','3%','4%','5%','6%','7%', ''])

ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Test Accuracy Boost')
ax.set_title('Test Accuracy Boost In Different Modalities')


plt.show()
pyplot.show()
