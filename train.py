import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image

data_dir = '/Users/derek/PycharmProjects/towhee_tmp/datasets'
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, index_file, transform=None):
        super().__init__()
        self.transform = transform
        self.data_dir = data_dir
        self.index_file = index_file
        # self.data_indices = datasets.ImageFolder(os.path.join(data_dir, transform))
        with open(index_file) as r_index_file:
            self.data_indices = r_index_file.readlines()

    def __len__(self):
        return len(self.data_indices)

    def __getitem__(self, index):
        img_path, label = self.data_indices[index].strip().split(':')
        img = Image.open(os.path.join(self.data_dir, img_path))
        if self.transform:
            img = self.transform

        return img, label


index_file = ''
train_dataset = MyDataset(data_dir, index_file, data_transforms['train'])
val_dataset = MyDataset(data_dir, index_file, data_transforms['val'])
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=4)

# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
