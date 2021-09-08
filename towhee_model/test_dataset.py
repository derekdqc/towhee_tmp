import unittest
import torch
import torchvision
from torchvision import transforms

from towhee_model.towhee_dataset import PyTorchImageDataset
from training_args import TrainingArguments
from trainer import Trainer


class DatasetTest(unittest.TestCase):
    def test_length(self):
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        dataset = PyTorchImageDataset('../kaggle_dataset/train', '../kaggle_dataset/labels.csv', data_transforms['train'])
        images, labels = dataset.images, dataset.labels
        self.assertEqual(len(images), len(labels))


if __name__ == '__main__':
    unittest.main(verbosity=1)
