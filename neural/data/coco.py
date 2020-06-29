import os

import numpy as np
from PIL import Image


from torch.utils.data import Dataset


class COCOStuff(Dataset):

    def __init__(self, root_dir, split='train', transforms=None):
        image_dir = os.path.join(root_dir, 'images', f'{split}2017')
        labels_dir = os.path.join(root_dir, 'stuff', f'{split}2017')
        ids = (fname[:-4] for fname in os.listdir(image_dir) if fname[-4:] == '.png')

        self.examples = [
            {'image': os.path.join(image_dir, f'{id}.jpg'),
             'mask': os.path.join(labels_dir, f'{id}.png'), }
            for id in ids
        ]

        self.transforms = transforms

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        image = example['image']
        mask = example['mask']

        image = Image.open(image).convert('RGB')
        image = np.asarray(image)

        mask = Image.open(mask)
        mask = np.asarray(mask)

        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            return augmented['image'], augmented['mask'].long()
        else:
            return image, mask
