import copy
import os
import time

import pandas as pd
import torch
import torchvision
from PIL import Image
from pandas import Series
from torch.utils.data import Dataset
from torchvision import transforms

from trainer import Trainer
from training_args import TrainingArguments

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

model = torchvision.models.resnet18(pretrained=False)


training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
)
trainer.train()
