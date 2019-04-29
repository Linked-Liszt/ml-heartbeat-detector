import pickle
import numpy as np
import math

MIN_FILTER = 250
MAX_FILTER = 450
LENGTH = 450

def pad_data(data):
    channel_1 = [sample[0] for sample in data]
    channel_2 = [sample[1] for sample in data]
    
    amount = int(math.floor((LENGTH-len(channel_1))/2))
    extra = 0
    if (((LENGTH-len(channel_1)) % 2) != 0):
        extra = 1

    pad_channel_1 = np.pad(channel_1, pad_width=(amount, amount+extra), mode='edge')
    pad_channel_2 = np.pad(channel_2, pad_width=(amount, amount+extra), mode='edge')
    return pad_channel_1, pad_channel_2

def normalize_data(channel_1, channel_2, interp_min, interp_max):
    channel_1_norm = np.interp(channel_1, (interp_min, interp_max), (0.0, 1.0))
    channel_2_norm = np.interp(channel_2, (interp_min, interp_max), (0.0, 1.0))
    return channel_1_norm, channel_2_norm

def reshape_heartbeat_data(channel_1, channel_2):
    full_output = []
    for i in range(len(channel_1)):
        full_output.append([channel_1[i],  channel_2[i]])
    return full_output

def get_min_max_interp(heartbeats):
    interp_min = float("inf")
    interp_max = float("-inf")
    for heartbeat in heartbeats:
        for sample in heartbeat:
            interp_min = min(interp_min, sample[0], sample[1])
            interp_max = max(interp_max, sample[0], sample[1])
    return interp_min, interp_max

with open('../../compressedDataFull/heartbeats.pickle', 'rb') as heartbeats_f:
    heartbeats = pickle.load(heartbeats_f)

with open('../../compressedDataFull/labels.pickle', 'rb') as labels_f:
    labels = pickle.load(labels_f)

unique_label_set = set(labels)
print("Unique Abnormal Labels:")
print(list(unique_label_set))

#FULL_ANNOTATIONS = ["N", "+", "P", "T", "~", "/", "M", "Q", " ", "|", 'J', 'j', 'x', 'R', 'f', 'L', 'E', 'a', 'A', 'V', ']', 'F', '!']
ANNOTATIONS = ['N', '+', '~', '!', ']', 'E', 'S', '/', 'L', 'Q', '|', 'F', 'j', 'a', 'R', 'A', 'J', 'x', 'f', 'V']

filtered_heartbeats = []
filtered_labels = []

print("Finding min and max for interp...")
interp_min, interp_max = get_min_max_interp(heartbeats)
print(interp_min)
print(interp_max)

print("Filtering Heartbeats...")
out_of_range = 0
for i in range(len(heartbeats)):

    heartbeat = heartbeats[i]

    if len(heartbeat) > MAX_FILTER or len(heartbeat) < MIN_FILTER:
        out_of_range += 1
    else:
        channel_1, channel_2 = pad_data(heartbeat)
        channel_1, channel_2 = normalize_data(channel_1, channel_2, interp_min, interp_max)
        filtered_heartbeats.append(reshape_heartbeat_data(channel_1, channel_2))
        
        #will throw error if not in annotations. That's good. 
        labelOutput = [0] * len(ANNOTATIONS)
        labelOutput[ANNOTATIONS.index(labels[i])] = 1
        filtered_labels.append(labelOutput)

print("Long/Short Heartbeats: " + str(out_of_range))

print("Filtered heartbeats and labels shape:")
print(np.shape(filtered_heartbeats))
print(np.shape(filtered_labels))
#print(filtered_labels[0])

print("Saving Data...")
np.save("../../compressedDataFull/labels_f1.npy", filtered_labels)
np.save("../../compressedDataFull/heartbeats_f1.npy", filtered_heartbeats)
print("Saved")
