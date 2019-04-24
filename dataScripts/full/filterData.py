import _pickle as pickle
import numpy as np
import math
import matplotlib.pyplot as plt

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

    full_output = []
    for i in range(len(pad_channel_1)):
        full_output.append([pad_channel_1[i],  pad_channel_2[i]])
    
    return full_output


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

out_of_range = 0
for i in range(len(heartbeats)):
    heartbeat = heartbeats[i]

    if len(heartbeat) > MAX_FILTER or len(heartbeat) < MIN_FILTER:
        out_of_range += 1
    else:
        filtered_heartbeats.append(pad_data(heartbeat))
        #will throw error if not in annotations. That's good. 
        filtered_labels.append(ANNOTATIONS.index(labels[i]))

print("Long/Short Heartbeats: " + str(out_of_range))

print(np.shape(filtered_heartbeats))
print(np.shape(filtered_labels))
print(filtered_labels[0])


with open('../../compressedDataFull/labels_f1.pickle', 'wb', protocal=pickle.HIGHEST_PROTOCOL) as labels_filtered_f:
    pickle.dump(filtered_labels, labels_filtered_f)

print("Created Labels File")

with open('../../compressedDataFull/heartbeats_f1.pickle', 'wb', protocal=pickle.HIGHEST_PROTOCOL) as heartbeats_filtered_f:
    pickle.dump(filtered_heartbeats, heartbeats_filtered_f)

print("Created Heartbeats File")

