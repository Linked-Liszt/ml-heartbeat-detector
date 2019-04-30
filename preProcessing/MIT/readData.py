import numpy as np
import os
import pickle
import wfdb
import wfdb.processing as wdpc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #outdated libraries in wfdb?

#Labels from here. Also list for refernce: https://github.com/MIT-LCP/wfdb-python/blob/master/wfdb/io/annotation.py
ANNOTATIONS = ["N", "+", "P", "T", "~", "/", "M", "Q", " ", "|", 'J', 'j', 'x', 'R', 'f', 'L', 'E', 'a', 'A', 'V', ']', 'F', '!']

def main():
    FILE_DIR = "../../rawData/MIT/"

    heartbeats_full = []
    labels_full = []

    for filename in os.listdir(FILE_DIR):
        if filename.endswith(".dat"):
            
            fn = FILE_DIR + filename[:-4]
            print(fn)

            sample, metadata = wfdb.rdsamp(fn)
            annotation = wfdb.rdann(fn, 'atr')

            lead1, lead2 = zip(*sample)
            xqrs = wdpc.XQRS(sig=np.asarray(lead1), fs=360)
            xqrs.detect()

            heartbeats, labels = split_samples_by_annotation(sample, annotation, xqrs.qrs_inds)

            heartbeats_full += heartbeats
            labels_full  += labels

            print(np.shape(heartbeats_full)) 
            print(np.shape(labels_full))

    with open('../../processedData/MIT/heartbeats.pickle', 'wb') as norm_file:
        pickle.dump(heartbeats_full, norm_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open('../../processedData/MIT/labels.pickle', 'wb') as abnorm_file:
        pickle.dump(labels_full, abnorm_file, protocol=pickle.HIGHEST_PROTOCOL)




def get_annotation_before(sample_point, annotation):
    if sample_point <= annotation.sample[0]:
        return annotation.symbol[0], annotation.sample[0]
    for i in range(len(annotation.sample)):
        if sample_point < annotation.sample[i]: #if equal we can go 1 over and go back
            return annotation.symbol[i-1], annotation.sample[i-1]
    return annotation.symbol[len(annotation.symbol)]

def get_annotation_after(sample_point, annotation):
    if (len(annotation.sample) != len(annotation.symbol)):
        raise Exception("bad annotation?")
    if sample_point >= annotation.sample[len(annotation.sample) - 2]: #need to catch if between next to last and last
        return annotation.symbol[len(annotation.symbol) - 1], annotation.sample[len(annotation.sample) - 1]
    for i in range(len(annotation.sample) - 1):
        if sample_point < annotation.sample[i]: #should always have 1 symbol ahead
            return annotation.symbol[i+1], annotation.sample[i+1]


def get_nearest_annotation(sample_point, annotation):
    ann_before, index_before = get_annotation_before(sample_point, annotation)
    ann_after, index_after = get_annotation_after(sample_point, annotation)
    if (abs(int(sample_point) - int(index_before)) < abs(int(sample_point) - int(index_after))):
        return ann_before
    else:
        return ann_after

def split_samples_by_annotation(sample, annotation, qrs_indexes):
    samples = []
    labels = []

    i = 2
    while (i < (len(qrs_indexes) - 2)): # skip first 2 and last 2
        qrs_index = int(qrs_indexes[i])

        ann = get_nearest_annotation(qrs_index, annotation)
        begin = int(int(qrs_indexes[i-1]) + ((qrs_index - int(qrs_indexes[i-1]))/2)) #half between next qrs
        end = int(int(qrs_indexes[i+1]) - ((int(qrs_indexes[i+1])-qrs_index)/2))
        
        samples.append(sample[begin:end])
        labels.append(ann)

        i += 1
    return samples, labels

if __name__ == '__main__':
    main()