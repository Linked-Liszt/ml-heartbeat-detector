import numpy as np
import math
import matplotlib.pyplot as plt

def get_min_max_interp(heartbeats):
    interp_min = float("inf")
    interp_max = float("-inf")
    for heartbeat in heartbeats:
        for sample in heartbeat:
            interp_min = min(interp_min, sample[0], sample[1])
            interp_max = max(interp_max, sample[0], sample[1])
    return interp_min, interp_max


heartbeats = np.load("../../processedData/MIT/heartbeats_f1.npy")
labels = np.load("../../processedData/MIT/labels_f1.npy")

min_h, max_h = get_min_max_interp(heartbeats)
print(min_h)
print(max_h)


for heartbeat in heartbeats:
    plt.plot([sample[0] for sample in heartbeat])
    plt.show()

