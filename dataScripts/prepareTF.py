import pickle
import numpy as np

with open('../compressedData/normal_f1.pickle', 'rb') as normal_f:
    normal_data = pickle.load(normal_f)

with open('../compressedData/abnormal_f1.pickle', 'rb') as abnormal_f:
    abnormal_data = pickle.load(abnormal_f)

normal_tf = []
for n_d in normal_data:
    normal_tf.append([n_d, 0])

abnormal_tf = []
for a_d in abnormal_data:
    abnormal_tf.append([a_d, 1])

print(np.shape(normal_tf))
print(np.shape(abnormal_tf))

with open('../compressedData/normal_f1_TF.pickle', 'wb') as norm_file:
        pickle.dump(normal_tf, norm_file)

with open('../compressedData/abnormal_f1_TF.pickle', 'wb') as abnorm_file:
    pickle.dump(abnormal_tf, abnorm_file)