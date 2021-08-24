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
    def __init__(self, image_path, label_file):
        self.image_path = image_path
        self.label_file = label_file

    def __getitem__(self, index):
        fn = self.images[index]
        img = self.loader(fn)
        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)

    @property
    def loader(self, image_path, data_transform):
        """
        read and preprocess one image
        """
        img_pil = Image.open(image_path)
        img_tensor = data_transform(img_pil)
        return img_tensor

    def read_label(self):
        df = pd.read_csv(self.label_file)
        images = Series.to_numpy(df["image_name"])
        # TODO: 这里的images需要和__getitem__关联
        return images