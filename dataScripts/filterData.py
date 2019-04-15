import pickle
import numpy as np
import math

MIN_FILTER = 250
MAX_FILTER = 450
LENGTH = 450

def pad_data(data):
    amount = int(math.floor((LENGTH-len(data))/2))
    extra = 0
    if (((LENGTH-len(data)) % 2) != 0):
        extra = 1
    return np.pad(data, pad_width=(amount, amount+extra), mode='edge')


with open('compressedData/normal.pickle', 'rb') as normal_f:
    normal_data = pickle.load(normal_f)

with open('compressedData/abnormal.pickle', 'rb') as abnormal_f:
    abnormal_data = pickle.load(abnormal_f)

with open('compressedData/abnormal_label.pickle', 'rb') as abnormal_l_f:
    abnormal_labels = pickle.load(abnormal_l_f)

unique_label_set = set(abnormal_labels)
print("Unique Abnormal Labels:")
print(list(unique_label_set))


filtered_normal = []
filtered_abnormal = []
filtered_abnormal_labels = []

n_ove500 = 0
for n_data in normal_data:
    if len(n_data) > MAX_FILTER or len(n_data) < MIN_FILTER:
        n_ove500 += 1
    else:
        filtered_normal.append(pad_data(n_data))


print("Long/Sort Normal " + str(n_ove500))
print("Total Normal Beats: " + str(len(normal_data)))

a_ove500 = 0
for i in range(len(abnormal_data)):
    a_data = abnormal_data[i]
    if len(a_data) > MAX_FILTER or len(a_data) < MIN_FILTER:
        a_ove500 += 1
    else:
        filtered_abnormal.append(pad_data(a_data))
        filtered_abnormal_labels.append(abnormal_labels[i])

print("Long/Short Abnormal " + str(a_ove500))
print("Total Abnormal Beats: " + str(len(abnormal_data)))

print(np.shape(filtered_normal))
print(np.shape(filtered_abnormal))
print(np.shape(filtered_abnormal_labels))

with open('compressedData/normal_f1.pickle', 'wb') as norm_file:
        pickle.dump(filtered_normal, norm_file)

with open('compressedData/abnormal_f1.pickle', 'wb') as abnorm_file:
    pickle.dump(filtered_abnormal, abnorm_file)

with open('compressedData/abnormal_label_f1.pickle', 'wb') as abnorm_label_file:
    pickle.dump(filtered_abnormal_labels, abnorm_label_file)