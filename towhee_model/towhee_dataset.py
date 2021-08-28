import pandas as pd
import numpy as np
from PIL import Image
from pandas import Series
from torch.utils.data import Dataset


class PytorchImageDataset(Dataset):
    """
        PytorchImageDataset is a dataset class for PyTorch.

        Args:
            image_path (:obj:`str`):
                Path to the images for your dataset.

            label_file (:obj:`Dict[str, str]`):
                 Path to your label file. The label file should be a csv file. The columns should be [image_name, category],
                'image_name' is the path of your image, 'category' is the label of accordance image. For example:
                [000bec180eb18c7604dcecc8fe0dba07.jpg, dog] for one row. Note that the first row should be[image_name, category]
        """

    def __init__(self, image_path, label_file, data_transform=None):
        self.image_path = image_path
        self.label_file = label_file
        self.data_transform = data_transform

    def __getitem__(self, index):
        df = pd.read_csv(self.label_file)
        self.images = Series.to_numpy(df['image_name'])
        self.labels = Series.to_numpy(df['category'])

        label = self.labels[index]
        fn = self.images[index]
        img = Image.open(fn)
        if self.data_transform:
            img = self.data_transform(img)

        return img, label

    def __len__(self):
        return len(self.images)
