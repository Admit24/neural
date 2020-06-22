from os import listdir
from os.path import join
from numpy import array
import cv2
from torch.utils.data import Dataset


class BDDSegmentation(Dataset):
    CLASSES = array([
        'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle',
    ])

    def __init__(self, root_dir, split='train', transforms=None):
        images_dir = join(root_dir, 'images', split)
        labels_dir = join(root_dir, 'labels', split)

        def generate_examples(images_dir, labels_dir):
            images = listdir(images_dir)
            images = [join(images_dir, image) for image in images]
            images = sorted(images)
            labels = listdir(labels_dir)
            labels = [join(labels_dir, image) for image in labels]
            labels = sorted(labels)

            assert len(images) == len(labels)

            for image, label in zip(images, labels):
                yield {
                    'image': image,
                    'label': label,
                }

        self.examples = list(generate_examples(images_dir, labels_dir))
        self.transforms = transforms if transforms is not None else dict

    def __getitem__(self, index):
        o = self.examples[index]

        image = o['image']
        label = o['label']

        image = cv2.imread(image, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = cv2.imread(label, cv2.IMREAD_GRAYSCALE)

        out = self.transforms(image=image, mask=label)
        return out['image'], out['mask'].long()

    def __len__(self):
        return len(self.examples)
