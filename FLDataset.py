import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv
import pandas as pd
import os
import cv2 
from itertools import combinations
import random
import collections
# from torchvision.transforms import ToTensor

def covidIID(dataset, num_users):
    images = int(len(dataset)/num_users)
    users_dict, indeces = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(i)
        users_dict[i] = set(np.random.choice(indeces, images, replace=False))
        indeces = list(set(indeces) - users_dict[i])

    return users_dict

def covidNonIID(dataset, num_users, c_num, noniid_c):
    classes, images = c_num, len(dataset)/c_num
    classes_indx = [i for i in range(classes)]
    users_dict = {i: np.array([]) for i in range(num_users)}
    indeces = np.arange(classes*images)
    unsorted_labels = dataset.get_labels()
    
    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_unsortedlabels = indeces_unsortedlabels.astype(int)
    shuffled_indices = np.random.permutation(len(indeces_unsortedlabels[0]))
    indeces_unsortedlabels[0] = indeces_unsortedlabels[0][shuffled_indices]
    indeces_unsortedlabels[1] = indeces_unsortedlabels[1][shuffled_indices]
    print("indeces_unsortedlabels ", indeces_unsortedlabels)
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]
    indeces_labels.astype(int)
    indeces.astype(int)
#     print(indeces_labels)
    
#     label list with index
    index_label = [[] for i in range(c_num)]
    for i in range(len(indeces_labels[1])):
        index_label[indeces_labels[1][i]].append(indeces_labels[0][i])
    print("index_label: ", index_label)
        
    client_classes = []
    comb = []
    for i in list(combinations(list(range(0,c_num)), noniid_c)):
        print(i)
        comb.append(i)
    print(comb)
    
    # classes of client
    for i in range(num_users):
        client_classes.append(comb[i%c_num])
    client_classes = np.array(client_classes)
    c = client_classes.flatten()
    print(client_classes)
    
    # count of labels
    label_count = collections.Counter(c)
    print(label_count)
    
#     for i in range(num_users):
#         np.random.seed(i)
#         temp = set(np.random.choice(classes_indx, 2, replace=False))
#         classes_indx = list(set(classes_indx) - temp)
#         for t in temp:
#             users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)
#     return users_dict

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
    train_dataset = CovidDataset('./train.csv', transform=transform)
    test_dataset = CovidDataset('./test.csv', transform=transform)
    train_group, test_group = None, None
    if iidtype == 'iid':
        train_group = covidIID(train_dataset, num_users)
        test_group = covidIID(test_dataset, num_users)
    elif iidtype == 'noniid':
        train_group = covidNonIID(train_dataset, num_users, c_num, noniid_c)
        test_group = covidNonIID(test_dataset, num_users, c_num, noniid_c)
    else:
        train_group = covidNonIIDUnequal(train_dataset, num_users)
        test_group = covidNonIIDUnequal(test_dataset, num_users)
    return train_dataset, test_dataset, train_group, test_group

class FedDataset(Dataset):
    def __init__(self, dataset, indx):
        self.dataset = dataset
        self.indx = [int(i) for i in indx]
        
    def __len__(self):
        return len(self.indx)
    
    def __getitem__(self, item):
#         images, label = self.dataset[self.indx[item]]
        images = self.dataset[self.indx[item]].get('image')
        label = self.dataset[self.indx[item]].get('label')
        return torch.tensor(images).clone().detach(), torch.tensor(label).clone().detach()
    
class CovidDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data_info = pd.read_csv(csv_path, header=None)
        self.transform = transform
                
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.data_info.iloc[idx, 0]
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        label = self.data_info.iloc[idx, 1]
        label = np.array([label])
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_labels(self):
        labels = []
        for i in range(len(self.data_info)):
            labels.append(self.data_info.iloc[i, 1])
        
        return labels
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        img = cv2.resize(image, (self.output_size, self.output_size))

        return {'image': img, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
#         print("image type: ", type(image))
#         print("image shape: ", image.shape)
        tensor_img = torch.from_numpy(image)
        tensor_img = tensor_img.unsqueeze(dim=0)
        tensor_img = tensor_img.type('torch.FloatTensor')
        tensor_lb = torch.from_numpy(label)
#         print("tensor_img shape: ", tensor_img.shape)
#         print("tensor_lb shape: ", tensor_lb.shape)
#         print("tensor_img type: ", type(tensor_img))
        return {'image': tensor_img,
                'label': tensor_lb}

def getActualImgs(dataset, indeces, batch_size):
    return DataLoader(FedDataset(dataset, indeces), batch_size=batch_size, shuffle=True)