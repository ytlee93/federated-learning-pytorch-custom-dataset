import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import csv
import pandas as pd
import os
import cv2 
from itertools import combinations
import random
import collections
# from torchvision.transforms import ToTensor

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def covidIID(dataset, num_users):
    images = int(len(dataset)/num_users)
    users_dict, indeces = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(i)
        users_dict[i] = set(np.random.choice(indeces, images, replace=False))
        indeces = list(set(indeces) - users_dict[i])
    
    return users_dict

def covidNonIID(dataset, num_users, c_num, noniid_c):
    classes, images = c_num, int(len(dataset)/c_num)
    classes_indx = [i for i in range(classes)]
    users_dict = {i: [] for i in range(num_users)}
    indeces = np.arange(classes*images)
    unsorted_labels = dataset.targets
    print("In covid non IID: unsorted labels = ", unsorted_labels)
    
    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    shuffled_indices = np.random.permutation(len(indeces_unsortedlabels[0]))
    print("In covid non IID: shuffled indices = ", shuffled_indices)
    indeces_unsortedlabels[0] = indeces_unsortedlabels[0][shuffled_indices]
    indeces_unsortedlabels[1] = indeces_unsortedlabels[1][shuffled_indices]
    print("In covid non IID: indeces_unsortedlabels ", indeces_unsortedlabels)
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    print("In covid non IID: indeces_labels ", indeces_labels)
    indeces = indeces_labels[0, :]
    indeces_labels.astype(int)
    indeces.astype(int)
#     print(indeces_labels)
    
    # label list with index
    index_label = [[] for i in range(c_num)]
    for i in range(len(indeces_labels[1])):
        index_label[indeces_labels[1][i]].append(indeces_labels[0][i])
#     print("index_label: ", index_label)
        
    client_classes = []
    comb = []
    for i in list(combinations(list(range(0,c_num)), noniid_c)):
        print(i)
        comb.append(i)
    print("comb ", comb)
    
    # classes of client
    for i in range(num_users):
        client_classes.append(comb[i%c_num])
    client_classes = np.array(client_classes)
    c = client_classes.flatten()
    print("client_classes ", client_classes)
    
    # count of labels
    label_count = collections.Counter(c)
    print("label count ", label_count)
    
    for i in range(len(label_count)):
        index_label[i] = split(index_label[i], label_count[i])
        index_label[i] = list(index_label[i])
        
    temp = []
#     users_dict = dict.fromkeys(range(10), [])
    print("users_dict ", users_dict)
    for i in range(len(client_classes)):
        for j in range(len(client_classes[i])):
            cur_cls = client_classes[i][j]
            temp = index_label[cur_cls].pop()
    #         print("temp ", temp)
            users_dict[i] = np.concatenate((users_dict[i], np.array(temp)), axis=0).astype(int)
    
    for i in range(len(users_dict)):
        users_dict[i] = set(users_dict[i])
    
    return users_dict

def covidNonIIDUnequal(dataset, num_users):
    classes, images = 1200, 50
    classes_indx = [i for i in range(classes)]
    users_dict = {i: np.array([]) for i in range(num_users)}
    indeces = np.arange(classes*images)
    unsorted_labels = dataset.train_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]

    min_cls_per_client = 1
    max_cls_per_client = 30

    random_selected_classes = np.random.randint(min_cls_per_client, max_cls_per_client+1, size=num_users)
    random_selected_classes = np.around(random_selected_classes / sum(random_selected_classes) * classes)
    random_selected_classes = random_selected_classes.astype(int)

    if sum(random_selected_classes) > classes:
        for i in range(num_users):
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, 1, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)

        random_selected_classes = random_selected_classes-1

        for i in range(num_users):
            if len(classes_indx) == 0:
                continue
            class_size = random_selected_classes[i]
            if class_size > len(classes_indx):
                class_size = len(classes_indx)
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)
    else:

        for i in range(num_users):
            class_size = random_selected_classes[i]
            np.random.seed(i)
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)

        if len(classes_indx) > 0:
            class_size = len(classes_indx)
            j = min(users_dict, key=lambda x: len(users_dict.get(x)))
            temp = set(np.random.choice(classes_indx, class_size, replace=False))
            classes_indx = list(set(classes_indx) - temp)
            for t in temp:
                users_dict[j] = np.concatenate((users_dict[j], indeces[t*images:(t+1)*images]), axis=0)

    return users_dict

def load_dataset(num_users, iidtype, transform, c_num, noniid_c = 0):
    data_path = "./FinalCovid19Dataset/train"
    train_dataset = datasets.ImageFolder(root=data_path,
                                            transform=transform)
    train_group, test_group = None, None
    if iidtype == 'iid':
        train_group = covidIID(train_dataset, num_users)

    elif iidtype == 'noniid':
        train_group = covidNonIID(train_dataset, num_users, c_num, noniid_c)

    else:
        train_group = covidNonIIDUnequal(train_dataset, num_users)
        
    return train_dataset, train_group

def getActualImgs(dataset, indices, batch_size):
    client_dataset = Subset(dataset, indices)
    return DataLoader(client_dataset, batch_size=batch_size, shuffle=True)