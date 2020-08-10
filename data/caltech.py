from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split
import numpy as np

from PIL import Image

import os
import os.path
import sys

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Caltech(VisionDataset):
    """Caltech dataset handler"""

    base_folder = '101_ObjectCategories'

    train_indices_filepath = 'train.txt'
    test_indices_filepath = 'test.txt'

    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(os.path.join(root, self.base_folder), transform=transform, target_transform=target_transform)

        self.split = split

        if self.split == 'train':
            indices_filepath = self.train_indices_filepath
        else:
            indices_filepath = self.test_indices_filepath

        self.images = []
        self.labels = []

        # Map label names to incremental index
        # {0: 'class_name_0', 1: 'class_name_1', ...}
        self.label_index = {}

        with open(os.path.join(root, indices_filepath), 'r') as f:
            incremental_index = -1

            for image_path in f:
                image_path = image_path.strip()
                label = image_path.split("/")[0]

                if not label == 'BACKGROUND_Google': # Filter out this class
                    if label not in self.label_index.values():
                        incremental_index += 1
                        self.label_index[incremental_index] = label

                    self.labels.append(incremental_index)
                    self.images.append(pil_loader(os.path.join(root, self.base_folder, image_path)))

    def __getitem__(self, index):
        """Access an element through its index.

        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        image = self.images[index]
        label = self.labels[index]

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.labels)

    def train_validation_split(self, val_size=None, random_state=None):
        """Split the training set into training and validation sets.
        
        Args:
            val_size (float): proportion of the validation set (between 0 and 1).
            random_state (int): random state for reproducibility of results.
        Returns:
            train_indices, val_indices: training and validation set indices.
        """

        assert split == 'train'

        train_indices, val_indices = train_test_split(np.arange(len(self.labels)), 
                                                      test_size=val_size,
                                                      stratify=self.labels,
                                                      random_state=random_state)
        
        return train_indices, val_indices