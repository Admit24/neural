from os import listdir
from os.path import join
from torch.utils.data import Dataset
from cv2 import imread


class Imagenet(Dataset):

    def __init__(self, root_dir, split='train', transforms=None):
        self.labels = read_labels(join(root_dir, 'imagenet_class_labels.txt'))
        self.label2idx = {value: idx for idx, value in enumerate(self.labels)}

        def generate_examples():
            if split == 'train':
                train_dir = join(root_dir, 'train')
                for cls in self.labels:
                    label_idx = self.label2idx[cls]
                    for img in listdir(join(train_dir, cls)):
                        yield {
                            'image': join(train_dir, cls, img),
                            'label': label_idx,
                        }
            elif split == 'val':
                val_labels = read_labels(
                    join(root_dir, 'imagenet_val_labels.txt'))
                val_dir = join(root_dir, 'val')
                image_files = sorted(listdir(val_dir))
                for label, img in zip(val_labels, image_files):
                    yield {
                        'image': join(val_dir, img),
                        'label': self.label2idx[label],
                    }
            else:
                raise ValueError()

        self.examples = list(generate_examples())

        self.transforms = transforms if transforms is not None else dict

    def __getitem__(self, idx):
        o = self.examples[idx]

        image = o['image']
        image = imread(image)
        image = image[..., ::-1]
        label = o['label']

        return self.transforms(image=image, label=label)

    def __len__(self):
        return len(self.examples)


def read_labels(filename):
    with open(filename) as f:
        labels = f.readlines()
    labels = [label.strip() for label in labels]
    return labels
