import pickle
import numpy as np
import scipy.ndimage.interpolation as spi
import matplotlib.pyplot as plt

with open('compressedData/normal_f1.pickle', 'rb') as normal_f:
    normal_data = pickle.load(normal_f)

with open('compressedData/abnormal_f1.pickle', 'rb') as abnormal_f:
    abnormal_data = pickle.load(abnormal_f)

with open('compressedData/abnormal_label_f1.pickle', 'rb') as abnormal_l_f:
    abnormal_labels = pickle.load(abnormal_l_f)

avgs_n = []

avgs_a = []

for n in normal_data:
    avgs_n.append(len(n))

for a in abnormal_data:
    avgs_a.append(len(a))

print(np.mean(avgs_n))
print(np.min(avgs_n))
print("-------------")
print(np.mean(avgs_a))
print(np.min(avgs_a))

#299 avg
#286 median
"""
ratio = 300/len(normal_data[0])

example = spi.zoom(normal_data[0], ratio)

plt.plot(normal_data[0])
plt.plot(example, 'r+')
"""