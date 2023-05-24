import os
from os import path
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF


class CacheClassLabel(data.Dataset):
    """
    A dataset wrapper that has a quick access to all labels of datasets.
    """
    def __init__(self, dataset):
        super(CacheClassLabel, self).__init__()
        self.dataset = dataset
        self.labels = torch.LongTensor(len(dataset)).fill_(-1)
        # Changed from (dataset.root, to ('datasets',
        label_cache_filename = path.join('datasets', str(type(dataset))+'_'+str(len(dataset))+'.pth')
        print(label_cache_filename)
        if path.exists(label_cache_filename):
            self.labels = torch.load(label_cache_filename)
        else:
            for i, ddata in enumerate(dataset):
                self.labels[i] = ddata[1]
            #os.makedirs(label_cache_filename, exist_ok=True)
            torch.save(self.labels, label_cache_filename)
        self.number_classes = len(torch.unique(self.labels))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img,target = self.dataset[index]
        return img, target


class AppendName(data.Dataset):
    """
    A dataset wrapper that also return the name of the dataset/task
    """
    def __init__(self, dataset, name, first_class_ind=0):
        super(AppendName,self).__init__()
        self.dataset = dataset
        self.name = name
        self.first_class_ind = first_class_ind  # For remapping the class index

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img,target = self.dataset[index]
        target = target + self.first_class_ind
        # print('*******************')
        # print(target)
        return img, target, self.name


class Subclass(data.Dataset):
    """
    A dataset wrapper that return the task name and remove the offset of labels (Let the labels start from 0)
    """
    def __init__(self, dataset, class_list, remap=True):
        '''
        :param dataset: (CacheClassLabel)
        :param class_list: (list) A list of integers
        :param remap: (bool) Ex: remap class [2,4,6 ...] to [0,1,2 ...]
        '''
        super(Subclass,self).__init__()
        assert isinstance(dataset, CacheClassLabel), 'dataset must be wrapped by CacheClassLabel'
        self.dataset = dataset
        self.class_list = class_list
        print("class list, type: ", class_list, type(class_list))
        self.remap = remap
        # Subset of indices with respect to class_list (subset of classes to keep :))
        self.indices = []
        for c in class_list:
            self.indices.extend((dataset.labels==c).nonzero().flatten().tolist())
            
        print("remap: ", remap)
       
        
        if remap:
            self.class_mapping = {c: i for i, c in enumerate(class_list)}
        else:
            self.class_mapping = {c: c for i, c in enumerate(class_list)}
            print("self.class mapping: ",self.class_mapping)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
       
        img,target = self.dataset[self.indices[index]]

        if True: #self.remap:
       
            raw_target = target.item() if isinstance(target,torch.Tensor) else target
            target = self.class_mapping[raw_target]
       
        return img, target


class Permutation(data.Dataset):
    """
    A dataset wrapper that permute the position of features
    """
    def __init__(self, dataset, permute_idx):
        super(Permutation,self).__init__()
        self.dataset = dataset
        self.permute_idx = permute_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img,target = self.dataset[index]
        shape = img.size()
        img = img.view(-1)[self.permute_idx].view(shape)
        return img, target


class Rotation(data.Dataset):
    """
    A dataset wrapper that permute the position of features
    """
    def __init__(self, dataset, rotate_angle):
        super(Rotation,self).__init__()
        self.dataset = dataset
        self.rotate_angle = int(rotate_angle)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, target = self.dataset[index]
        rotated = TF.rotate(img, self.rotate_angle)
        return rotated, target


class Storage(data.Dataset):
    """
    A dataset wrapper used as a memory to store the datasets
    """
    def __init__(self):
        super(Storage, self).__init__()
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, index):
        return self.storage[index]

    def append(self,x):
        self.storage.append(x)

    def extend(self,x):
        self.storage.extend(x)