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
print(df.info())
print(df.head())
breed = df['breed']
breed_np = Series.to_numpy(breed)
print(type(breed_np))
print(breed_np.shape)  # (10222,)

# 看一下一共多少不同种类
breed_set = set(breed_np)
print(len(breed_set))  # 120

# 构建一个编号与名称对应的字典，以后输出的数字要变成名字的时候用：
breed_120_list = list(breed_set)
dic = {}
for i in range(120):
    dic[breed_120_list[i]] = i

file = Series.to_numpy(df["id"])
print(file.shape)

file = [i + ".jpg" for i in file]
file = [os.path.join("./kaggle_dataset/train", i) for i in file]
file_train = file[:8000]
file_test = file[8000:]
print(file_train)

np.save("./kaggle_dataset/file_train.npy", file_train)
np.save("./kaggle_dataset/file_test.npy", file_test)

breed = Series.to_numpy(df["breed"])
print(breed.shape)
number = []
for i in range(10222):
    number.append(dic[breed[i]])
number = np.array(number)
number_train = number[:8000]
number_test = number[8000:]
np.save("./kaggle_dataset/number_train.npy", number_train)
np.save("./kaggle_dataset/number_test.npy", number_test)

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


def default_loader(path):
    img_pil = Image.open(path)
    img_pil = img_pil.resize((224, 224))
    img_tensor = preprocess(img_pil)
    return img_tensor


# 当然出来的时候已经全都变成了tensor
class trainset(Dataset):
    def __init__(self, loader=default_loader):
        # 定义好 image 的路径
        self.images = file_train
        self.target = number_train
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)


class valset(Dataset):
    def __init__(self, loader=default_loader):
        # 定义好 image 的路径
        self.images = file_test
        self.target = number_test
        self.loader = loader

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        target = self.target[index]
        return img, target

    def __len__(self):
        return len(self.images)


train_data = trainset()
val_data = valset()
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
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)