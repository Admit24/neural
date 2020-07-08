from albumentations.pytorch import ToTensorV2 as ToTensor
import albumentations as albu
import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.metrics import Accuracy, IoU
from pytorch_lightning.callbacks import ModelCheckpoint
from neural.models.segmentation.lednet import LedNet
from neural.losses import OHEMLoss
from neural.optim.lr_scheduler import PolyLR

import argparse

import cv2
from neural.data.cityscapes import Cityscapes


class LednetModel(LightningModule):

    def __init__(self, learning_rate, total_steps):
        super().__init__()

        self.learning_rate = learning_rate
        self.total_steps = total_steps

        self.model = LedNet(3, 19)
        self.loss_fn = OHEMLoss(ignore_index=255, numel_frac=1 / 16)

        self.accuracy = Accuracy(num_classes=19)
        self.iou = IoU(num_classes=19)

    def forward(self, input):
        return self.model(input)

    def training_step(self, batch, batch_nb):
        x, y = batch

        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)

        log = {'train_loss': loss}

        return {'loss': loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        acc = self.accuracy(logits, y)
        iou = self.iou(logits, y)
        return {'accuracy': acc, 'miou': iou}

    def validation_epoch_end(self, outputs):
        acc = torch.stack([x['accuracy'] for x in outputs]).mean() / 2
        miou = torch.stack([x['miou'] for x in outputs]).mean() / 2
        return {'accuracy': acc,
                'miou': miou,
                'log': {'val_accuracy': acc, 'val_miou': miou}}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4,
        )
        scheduler = PolyLR(optimizer, self.learning_rate, total_steps=self.total_steps, gamma=0.9)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'interval': 'iteration',
        }


train_tfms = albu.Compose([
    albu.RandomScale([0.5, 2.0], interpolation=cv2.INTER_CUBIC, p=1.),
    albu.RandomCrop(768, 768),
    albu.HorizontalFlip(),
    albu.HueSaturationValue(),
    albu.Normalize(),
    ToTensor(),
])

val_tfms = albu.Compose([
    albu.Normalize(),
    ToTensor(),
])

train_dataset = Cityscapes('/srv/datasets/cityscapes', transforms=train_tfms)
val_dataset = Cityscapes('/srv/datasets/cityscapes', split='val', transforms=val_tfms)
train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=8, batch_size=8)
val_loader = torch.utils.data.DataLoader(val_dataset, num_workers=8, batch_size=1)

max_epochs = 20

model = LednetModel(
    learning_rate=5e-4,
    total_steps=len(train_loader) * max_epochs)

checkpoint_callback = ModelCheckpoint(
    filepath=None,
    save_top_k=1,
    monitor='miou',
    mode='max',
)

parser = argparse.ArgumentParser()
parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

trainer = Trainer.from_argparse_args(args,
                                     check_val_every_n_epoch=1,
                                     checkpoint_callback=checkpoint_callback)
trainer.fit(model, train_loader, val_loader)
