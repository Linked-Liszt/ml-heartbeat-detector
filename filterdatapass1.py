import pickle
import numpy as np

with open('normal.pickle', 'rb') as normal_f:
    normal_data = pickle.load(normal_f)

with open('abnormal.pickle', 'rb') as abnormal_f:
    abnormal_data = pickle.load(abnormal_f)

with open('abnormal_label.pickle', 'rb') as abnormal_l_f:
    abnormal_labels = pickle.load(abnormal_l_f)

unique_label_set = set(abnormal_labels)
print("Unique Abnormal Labels:")
print(list(unique_label_set))


filtered_normal = []
filtered_abnormal = []
filtered_abnormal_labels = []

n_ove500 = 0
for n_data in normal_data:
    if len(n_data) > 450:
        n_ove500 += 1
    else:
        filtered_normal.append(n_data)


print("Long Normal " + str(n_ove500))
print("Total Normal Beats: " + str(len(normal_data)))

a_ove500 = 0
for i in range(len(abnormal_data)):
    a_data = abnormal_data[i]
    if len(a_data) > 450:
        a_ove500 += 1
    else:
        filtered_abnormal.append(a_data)
        filtered_abnormal_labels.append(abnormal_labels[i])

print("Long Abnormal " + str(a_ove500))
print("Total Abnormal Beats: " + str(len(abnormal_data)))

print(np.shape(filtered_normal))
print(np.shape(filtered_abnormal))
print(np.shape(filtered_abnormal_labels))

with open('normal_f1.pickle', 'wb') as norm_file:
        pickle.dump(filtered_normal, norm_file)

with open('abnormal_f1.pickle', 'wb') as abnorm_file:
    pickle.dump(filtered_abnormal, abnorm_file)

with open('abnormal_label_f1.pickle', 'wb') as abnorm_label_file:
    pickle.dump(filtered_abnormal_labels, abnorm_label_file)