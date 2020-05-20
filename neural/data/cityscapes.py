from os import listdir
from os.path import join
import cv2
from numpy import array
from torch.utils.data import Dataset


class Cityscapes(Dataset):
    CLASSES = array([
        'unlabeled', 'ego vehicle', 'rectification border', 'out of roi',
        'static', 'dynamic', 'ground', 'road', 'sidewalk', 'parking',
        'rail track', 'building', 'wall', 'fence', 'guard rail',
        'bridge', 'tunnel', 'pole', 'polegroup', 'traffic light',
        'traffic sign', 'vegetation', 'terrain', 'sky', 'person',
        'rider', 'car', 'truck', 'bus', 'caravan', 'trailer', 'train',
        'motorcycle', 'bicycle', 'license plate'])

    TRAIN_MAPPING = array([
        255, 255, 255, 255, 255, 255, 255,   0,  1, 255, 255,   2,  3,  4, 255,
        255, 255,   5, 255,   6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 255,
        255,  16, 17, 18, 255])

    def __init__(self, root_dir, split='train', type='semantic', transforms=None):

        self.type = type

        def generate_examples(images_dir, labels_dir, annotation='fine'):
            labels_type = 'gtFine' if annotation == 'fine' else 'gtCoarse'
            cities = listdir(images_dir)
            for city in cities:
                city_dir = join(images_dir, city)

                for f in listdir(city_dir):
                    id = f[:-16]
                    image = join(images_dir, city, f)
                    semantic = join(labels_dir, city,
                                    f'{id}_{labels_type}_labelIds.png')
                    instance = join(labels_dir, city,
                                    f'{id}_{labels_type}_instanceIds.png')

                    if type == 'semantic':
                        yield {'image': image, 'label': semantic}
                    elif type == 'instance':
                        yield {'image': image, 'label': instance}
                    elif type == 'panotic':
                        yield {
                            'image': image,
                            'semantic': semantic,
                            'instance': instance,
                        }

        if type(split) is not list:
            split = [split]

        self.examples = []
        if 'train' in split:
            self.examples += list(generate_examples(
                join(root_dir, 'leftImg8bit', 'train'),
                join(root_dir, 'gtFine', 'train'),
                'fine'
            ))
        elif 'valid' in split:
            self.examples += list(generate_examples(
                join(root_dir, 'leftImg8bit', 'val'),
                join(root_dir, 'gtFine', 'val'),
                'fine'
            ))
        elif split in 'trainextra':
            self.examples += list(generate_examples(
                join(root_dir, 'leftImg8bit', 'train_extra'),
                join(root_dir, 'gtCoarse', 'train_extra'),
                'coarse'
            ))

        self.transforms = transforms if transforms is not None else dict

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        image = example['image']
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.type == 'semantic':
            label = cv2.imread(example['label'], cv2.IMREAD_GRAYSCALE)
            label = self.TRAIN_MAPPING[label]
            return self.transforms(image=image, label=label)
        elif self.type == 'instance':
            label = cv2.imread(example['label'], cv2.IMREAD_GRAYSCALE)
            return self.transforms(image=image, label=label)
        elif self.type == 'panotic':
            semantic = cv2.imread(example['semantic'], cv2.IMREAD_GRAYSCALE)
            semantic = self.TRAIN_MAPPING[semantic]
            instance = cv2.imread(example['instance'], cv2.IMREAD_GRAYSCALE)
            return self.transforms(
                image=image,
                semantic=sematic,
                instance=instance)
        else:
            raise RuntimeError("Invalid cityscapes dataset type.")
