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


class PytorchImageDataset(Dataset):
    """
        PytorchImageDataset is a dataset class for PyTorch.

        Args:
            image_path (:obj:`str`):
                Path to the images for your dataset.

            label_file (:obj:`Dict[str, str]`):
                 Path to your label file. The label file should be a csv file. The columns should be [image_name, category],
                'image_name' is the path of your image, 'category' is the label of accordance image. For example:
                [image_name, dog] for one row. Note that the first row should be[image_name, category]
        """

    def __init__(self, image_path, label_file, data_transform=None):
        self.image_path = image_path
        self.label_file = label_file
        self.data_transform = data_transform

        df = pd.read_csv(self.label_file)
        image_names = Series.to_numpy(df['image_name'])
        images = [i + ".jpg" for i in image_names]
        self.images = [os.path.join(self.image_path, i) for i in images]

        categories = Series.to_numpy(df['category'])
        # 看一下一共多少不同种类
        breed_set = set(categories)
        # 构建一个编号与名称对应的字典，以后输出的数字要变成名字的时候用：
        breed_list = list(breed_set)
        dic = dict()
        for i in range(len(breed_list)):
            dic[breed_list[i]] = i
        self.labels = [dic[categories[i]] for i in range(len(categories))]

    def __getitem__(self, index):
        label = self.labels[index]
        fn = self.images[index]
        img = Image.open(fn)
        if self.data_transform:
            img = self.data_transform(img)

        return img, label

    def __len__(self):
        return len(self.labels)


train_data = PytorchImageDataset('./kaggle_dataset/train', './kaggle_dataset/labels.csv', data_transforms['train'])

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
