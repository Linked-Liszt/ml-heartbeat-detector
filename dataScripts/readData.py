import wfdb
import wfdb.processing as wdpc
import numpy as np
import os
import pickle
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #outdated libraries?

#Labels from here: https://github.com/MIT-LCP/wfdb-python/blob/master/wfdb/io/annotation.py
ANNOTATIONS = ["N","+", "P", "T", "~", "/", "M", "Q", " ", "|", 'J', 'j', 'x', 'R', 'f', 'L', 'E', 'a', 'A', 'V', ']', 'F', '!']

def main():
    FILE_DIR = "data/"

    heartbeats_full = []
    labels_full = []

    for filename in os.listdir(FILE_DIR):
        if filename.endswith(".dat"):
            
            fn = FILE_DIR + filename[:-4]
            print(fn)

            sample, _ = wfdb.rdsamp(fn)
            annotation = wfdb.rdann(fn, 'atr')

            sample_array, _ = zip(*sample)
            xqrs = wdpc.XQRS(sig=np.asarray(sample_array), fs=360)
            xqrs.detect()

            heartbeats, labels = split_samples_by_annotation(sample_array, annotation, xqrs.qrs_inds)

            heartbeats_full.append(heartbeats)
            labels_full.append(labels)

            print(np.shape(heartbeats_full))
            print(np.shape(labels_full))
    
    with open('normal.pickle', 'wb') as norm_file:
        pickle.dump(normal_final, norm_file)

    with open('abnormal.pickle', 'wb') as abnorm_file:
        pickle.dump(abnormal_final, abnorm_file)

    with open('abnormal_label.pickle', 'wb') as abnorm_label_file:
        pickle.dump(abnormal_label_final, abnorm_label_file)
    
    
    #print(np.shape(a))
    #print(sample[1])

    #print(xqrs.qrs_inds)
    #wfdb.plot_wfdb(record=record, annotation=annotation)
    #wfdb.plot_items(signal=sample, ann_samp=[xqrs.qrs_inds])
    #print(np.shape(record[0]))
    #print(record[1])
    #print(annotation.sample)
    #print(annotation.symbol)
    #print(len(record.p_signal))

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