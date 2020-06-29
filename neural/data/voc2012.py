import os

import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class VOC2012Segmentation(Dataset):
    """The Pascal VOC 2012 Dataset
    """

    def __init__(self, root_dir, split='train', transforms=None):
        split_ids_file = os.path.join(
            root_dir, 'ImageSets', 'Segmentation', split + '.txt')
        split_ids = open(split_ids_file).readlines()
        split_ids = [id.strip() for id in split_ids]

        images_dir = os.path.join(root_dir, 'JPEGImages')
        labels_dir = os.path.join(root_dir, 'SegmentationClass')

        self.examples = [
            {'image': os.path.join(images_dir, id + '.jpg'),
             'mask': os.path.join(labels_dir, id + '.png')}
            for id in split_ids
        ]

        self.transforms = transforms

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        image = example['image']
        mask = example['mask']

        image = Image.open(image)
        image = np.asarray(image)

        mask = Image.open(mask)
        mask = np.asarray(mask)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            return augmented['image'], augmented['mask'].long()
        else:
            return image, mask
