import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle as skshuffle

heartbeats = np.load("../../processedData/MIT/heartbeats_f1.npy")
labels = np.load("../../processedData/MIT/labels_f1.npy")

ANNOTATIONS = ["N", "+", "P", "T", "~", "/", "M", "Q", "NA", "|", 'J', 'j', 'x', 'R', 'f', 'L', 'E', 'a', 'A', 'V', ']', 'F', '!']

CHART_ROWS = 10
CHART_COLUMNS = 20

shuffled_heartbeats, shuffled_labels = skshuffle(heartbeats, labels)

fig, axs =plt.subplots(CHART_ROWS, CHART_COLUMNS)

print(np.shape(shuffled_heartbeats))


i = 0
for row in axs:
    for j in range(CHART_COLUMNS):
        #reshape
        channel1 = []
        channel2 = []
        for k in range(len(shuffled_heartbeats[i+j])):
            channel1.append(shuffled_heartbeats[i+j][k][0])
            channel2.append(shuffled_heartbeats[i+j][k][1])

        row[j].plot(channel1)
        row[j].plot(channel2)
        row[j].get_xaxis().set_visible(False)
        row[j].get_yaxis().set_visible(False)

        #find label
        found_annotation = "unknown"
        for l in range(len(shuffled_labels[i+j])):
            if shuffled_labels[i+j][l] == 1:
                found_annotation = ANNOTATIONS[l]
                break

        row[j].set_title(found_annotation)
        i += 1
    i += 1
plt.show()
