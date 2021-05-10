import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import csv
import pandas as pd
import os
import cv2 
# from torchvision.transforms import ToTensor

def covidIID(dataset, num_users):
    images = int(len(dataset)/num_users)
    users_dict, indeces = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        np.random.seed(i)
        users_dict[i] = set(np.random.choice(indeces, images, replace=False))
        indeces = list(set(indeces) - users_dict[i])
    return users_dict

def covidNonIID(dataset, num_users):
    classes, images = 200, 300
    classes_indx = [i for i in range(classes)]
    users_dict = {i: np.array([]) for i in range(num_users)}
    indeces = np.arange(classes*images)
    unsorted_labels = dataset.train_labels.numpy()

    indeces_unsortedlabels = np.vstack((indeces, unsorted_labels))
    indeces_labels = indeces_unsortedlabels[:, indeces_unsortedlabels[1, :].argsort()]
    indeces = indeces_labels[0, :]

    for i in range(num_users):
        np.random.seed(i)
        temp = set(np.random.choice(classes_indx, 2, replace=False))
        classes_indx = list(set(classes_indx) - temp)
        for t in temp:
            users_dict[i] = np.concatenate((users_dict[i], indeces[t*images:(t+1)*images]), axis=0)
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

def load_dataset(num_users, iidtype):
    train_dataset = CovidDataset('./train.csv', transform=transforms.Compose([Rescale(64), ToTensor()]))
    test_dataset = CovidDataset('./test.csv', transform=transforms.Compose([Rescale(64), ToTensor()]))
    train_group, test_group = None, None
    if iidtype == 'iid':
        train_group = covidIID(train_dataset, num_users)
        test_group = covidIID(test_dataset, num_users)
    elif iidtype == 'noniid':
        train_group = covidNonIID(train_dataset, num_users)
        test_group = covidNonIID(test_dataset, num_users)
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
        tensor = torch.from_numpy(image)
        tensor = tensor.unsqueeze(dim=0)
        return {'image': tensor,
                'label': torch.from_numpy(label)}

def getActualImgs(dataset, indeces, batch_size):
    return DataLoader(FedDataset(dataset, indeces), batch_size=batch_size, shuffle=True)