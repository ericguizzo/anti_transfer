import random
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt



train = [0.6996790754683,
0.74166822186927,
0.756309865900383,
0.73045205406556,
0.751537622392508,
0.721696732652192,
0.729989623243934,
0.720537196679438,
0.681944444444444,
0.69011095146871,
0.706027298850575,
0.733242071094083,
0.72136254789272]

test =[0.619239672364672,
0.628846153846154,
0.637179487179487,
0.636431623931624,
0.642735042735043,
0.642307692307692,
0.636217948717949,
0.638247863247863,
0.615277777777778,
0.624145299145299,
0.637286324786325,
0.63215811965812,
0.636004273504274]


train_libri_random = np.array([
0.725185185185185,
0.784814814814815,
0.759259259259259,
0.746666666666667,
0.748888888888889,
0.737592592592593,
0.759814814814815,
0.736851851851852,
0.702222222222222,
0.713148148148148,
0.698518518518519,
0.72037037037037,
0.706296296296296])

train_libri_actor = np.array([
0.729688850475367,
0.733578219533276,
0.7286269430051814,
0.7252806563039723,
0.6798359240069084,
0.7413644214162349,
0.6867443868739207,
0.6279145077720207,
0.680915371329879,
0.7222582037996544,
0.7140544041450777,
0.7900474956822108,
0.6720639032815198])

train_iemo_actor = np.array([
0.6471477960242,
0.752909482758621,
0.752262931034483,
0.744504310344828,
0.750538793103448,
0.744719827586207,
0.74385775862069,
0.741056034482759,
0.6625,
0.671443965517241,
0.733081896551724,
0.742133620689655,
0.734375
])

train_iemo_random = np.array([
0.696694470188446,
0.69537037037037,
0.757407407407407,
0.700185185185185,
0.755185185185185,
0.682777777777778,
0.686296296296296,
0.683703703703704,
0.681111111111111,
0.685740740740741,
0.686481481481482,
0.737222222222222,
0.687777777777778
])


test_libri_random = np.array([
0.668910256410256,
0.668910256410257,
0.656089743589744,
0.677884615384616,
0.661217948717949,
0.668910256410257,
0.660897435897436,
0.663782051282051,
0.648397435897436,
0.675320512820513,
0.667628205128205,
0.658653846153846,
0.644230769230769
])

test_libri_actor = np.array([
0.602564102564102,
0.60448717948718,
0.6033591731266149,
0.6040051679586563,
0.599483204134367,
0.5891472868217055,
0.6111111111111112,
0.5846253229974161,
0.6201550387596899,
0.5988372093023255,
0.5930232558139535,
0.603359173126615,
0.6072351421188631
])

test_iemo_random = np.array([
0.63048433048433,
0.640705128205128,
0.632371794871795,
0.653205128205128,
0.665064102564102,
0.644551282051282,
0.640705128205128,
0.640705128205128,
0.63974358974359,
0.634294871794872,
0.641025641025641,
0.632051282051282,
0.640705128205128
])

test_iemo_actor = np.array([
0.575,
0.601282051282051,
0.623076923076923,
0.578205128205128,
0.601923076923077,
0.613461538461539,
0.607051282051282,
0.61025641025641,
0.557692307692308,
0.562820512820513,
0.603205128205128,
0.605769230769231,
0.623076923076923,
])
train_random_baseline = 0.6905
train_actor_baseline = 0.678
test_random_baseline = 0.637
test_actor_baseline = 0.572

#print (train_iemo_random)
train_iemo_random = [x-train_random_baseline for x in train_iemo_random]
test_iemo_random = [x-test_random_baseline for x in test_iemo_random]
train_libri_random = [x-train_random_baseline for x in train_libri_random]
test_libri_random = [x-test_random_baseline for x in test_libri_random]

train_iemo_actor = [x-train_actor_baseline for x in train_iemo_actor]
test_iemo_actor = [x-test_actor_baseline for x in test_iemo_actor]
train_libri_actor = [x-train_actor_baseline for x in train_libri_actor]
test_libri_actor = [x-test_actor_baseline for x in test_libri_actor]

train_mean = np.mean([train_iemo_random,train_iemo_actor,train_libri_random,train_libri_actor],axis=0)
test_mean = np.mean([test_iemo_random,test_iemo_actor,test_libri_random,test_libri_actor],axis=0)
#print (train_iemo_random)


train

overall_train = ((75.5-69) + (74.5-67.8) + (93.9-91.8) + (36.4-42.2)) / 4
print ('culo', overall_train)

#test
sound_goodness =((86.3-83.8) + (34.3-22.8) ) / 2
ser = ((66.5-63.7) + (61.3-57.2) ) / 2
overall = ((86.3-83.8) + (34.3-22.8) + (66.5-63.7) + (61.3-57.2)) / 4
bigger_pre = ((86.3-83.8) + (30-22.8) + (66.9-63.7) + (61.1-57.2)) / 4
same_pre = ((66.5-63.7) + (61.3-57.1) + (85.7-83.8) + (34.3-22.8))/4


print(sound_goodness,ser,overall,bigger_pre,same_pre)
sound_goodness = 7.0
ser = 3.45
overall = 5.25
bigger_pre = 4.2
same_pre = 5.1



# Fixing random state for reproducibility
np.random.seed(19680801)

plt.rcdefaults()
fig, ax = plt.subplots()

# Example data
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
