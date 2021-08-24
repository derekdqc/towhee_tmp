import time
import copy
import pandas as pd
import numpy as np
import os
import torch
import torchvision
from torch import nn, optim
from torch.optim import lr_scheduler

from PIL import Image
from pandas import Series, DataFrame
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

df = pd.read_csv('./kaggle_dataset/labels.csv')
breed = df['breed']
breed_np = Series.to_numpy(breed)

# 看一下一共多少不同种类
breed_set = set(breed_np)
# 构建一个编号与名称对应的字典，以后输出的数字要变成名字的时候用：
breed_120_list = list(breed_set)
dic = {}
for i in range(120):
    dic[breed_120_list[i]] = i

file = Series.to_numpy(df["id"])

file = [i + ".jpg" for i in file]
file = [os.path.join("./kaggle_dataset/train", i) for i in file]
file_train = file[:8000]
file_val = file[8000:]

breed = Series.to_numpy(df["breed"])
label = []
for i in range(10222):
    label.append(dic[breed[i]])
# label = np.array(label)
label_train = label[:8000]
label_val = label[8000:]
print('label_train: ', label_train)

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


def train_loader(path):
    img_pil = Image.open(path)
    img_tensor = data_transforms['train'](img_pil)
    return img_tensor


def val_loader(path):
    img_pil = Image.open(path)
    img_tensor = data_transforms['val'](img_pil)
    return img_tensor


class MyDataset(Dataset):
    def __init__(self, images, labels, loader):
        # 定义好 image 的路径
        self.images = images
        self.labels = labels
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)


train_data = MyDataset(file_train, label_train, train_loader)
val_data = MyDataset(file_val, label_val, val_loader)
train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4, shuffle=True)

dataloaders = {'train': train_loader,
               'val': val_loader}
dataset_sizes = {'train': len(train_data),
                 'val': len(val_data)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


model_ft = torchvision.models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 120)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)