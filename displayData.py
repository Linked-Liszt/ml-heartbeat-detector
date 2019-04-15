import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('abnormal_f1.pickle', 'rb') as normal_f:
    normal_data = pickle.load(normal_f)

fig, axs =plt.subplots(20, 20)

i = 0
for row in axs:
    for i2 in range(20):
        row[i2].plot(normal_data[i+i2])
        row[i2].get_xaxis().set_visible(False)
        row[i2].get_yaxis().set_visible(False)
        i += 1
    i += 1
plt.show()