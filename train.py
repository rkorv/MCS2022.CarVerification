from torch.utils.data import DataLoader, ConcatDataset
import torch
import pytorch_lightning as pl

import src.models.classifier
import src.datasets.CarsDataset
import src.utils.transforms
import src.utils.losses
import src.utils.vis


class CarsReidentificationTrainEffnet(pl.LightningModule):
    lr = 0.00006
    EMBEDDING_DIM = 2048

    def __init__(self, num_classes=5540, pretain_path=None, model_name="m"):
        super().__init__()

        self.net = src.models.classifier.EffNetv2(
            num_classes, self.EMBEDDING_DIM, model_name=model_name
        )

        if pretain_path is not None:
            self.net.load_state_dict(torch.load(pretain_path))

        self.loss = src.utils.losses.ReIdentificationLossWithClassification()

    def calc_loss(self, out, mask, prefix="Train"):
        loss_val, loss_stat = self.loss(out[0], out[1], mask)
        pub_loss_stat = {f"{prefix}/{k}": v for k, v in loss_stat.items()}

        on_step = False if prefix == "Val" else True
        on_epoch = not on_step

        self.log_dict(pub_loss_stat, on_step=on_step, on_epoch=on_epoch, add_dataloader_idx=False)
        return loss_val

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_nb):
        imgs, labels = batch[0], batch[1]
        imgs = torch.concat(imgs, axis=0).float()
        labels = torch.concat(labels, axis=0).long()

        out = self.forward(imgs)
        return {"loss": self.calc_loss(out, labels)}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0.05
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
        return (
            {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                },
            },
        )


if __name__ == "__main__":
    MODEL_RESUME = None
    PRETRAIN = None

    DATASET_ROOT_DIR = "../datasets"
    LOGS_DIR = "./logs"

    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    TARGET_SHAPE = (300, 300)
    BATCH_SIZE = 4
    AUGS = True

    train_transforms = src.utils.transforms.CarsTransforms(
        TARGET_SHAPE, IMAGE_MEAN, IMAGE_STD, augs=AUGS
    )
    trainset = ConcatDataset(
        [
            src.datasets.CarsDataset.FullDatasetBalanced(
                DATASET_ROOT_DIR,
                transforms=train_transforms,
                triplet=True,
                augs=AUGS,
            ),
        ]
    )

    num_classes = trainset.datasets[0].CLASS_NUM
    print("Num classes: ", num_classes)

    model = CarsReidentificationTrainEffnet(num_classes, PRETRAIN, model_name="m")

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    trainer_cfg = {
        "gpus": 1,
        "logger": pl.loggers.TensorBoardLogger(save_dir=LOGS_DIR),
        "precision": 16,
        "auto_lr_find": True,
        "accumulate_grad_batches": {0: 1, 2: 2, 4: 3},
        "max_epochs": 8,
        "callbacks": [
            pl.callbacks.ModelCheckpoint(dirpath=LOGS_DIR, every_n_train_steps=2000, save_top_k=-1),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
    }

    trainer = pl.Trainer(**trainer_cfg)
    _ = trainer.fit(
        model,
        train_dataloaders=train_loader,
        ckpt_path=MODEL_RESUME,
    )
