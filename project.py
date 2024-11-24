from pathlib import Path

import torchmetrics
import torchvision.models as models
import lightning as L
from matplotlib import pyplot as plt

from utils import ImageFolder, create_train_test_split

import torch
import torch.nn as nn
import torchinfo
from lightning.pytorch.loggers import TensorBoardLogger
from torchvision import transforms
from torchvision.transforms.v2 import functional as TF
from utils import DataSetModule
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

SCRIPT_DIR = Path(__file__).parent
DATA_BASE_PATH = SCRIPT_DIR / "data" / "data_set"

# Model
def get_resnet18_model_and_transforms(num_classes):
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Sequential(nn.Linear(512, num_classes))


    tr_train = transforms.Compose(
        [
            transforms.v2.RGB(),
            transforms.RandomResizedCrop((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    tr_val = transforms.Compose(
        [
            transforms.v2.RGB(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return net, tr_train, tr_val



# Classifier
class Classifier(L.LightningModule):
    def __init__(self, model, num_classes=2, learning_rate=0.001, weight_decay: float = 0.0):
        super().__init__()
        self.model = model

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.loss_fn = nn.CrossEntropyLoss()

        # Accuracy
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="multiclass", num_classes=num_classes
        )

        # Loss
        self.train_loss = torchmetrics.MeanMetric()
        self.val_loss = torchmetrics.MeanMetric()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update all metrics
        self.train_loss.update(loss)
        self.train_accuracy.update(preds, y)

        # Log metrics for bartch
        self.log("train/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)

        self.log(
            "train/acc_step",
            self.train_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def on_train_epoch_end(self):
        # Log average loss and metrics for the entire epoch
        avg_loss = self.train_loss.compute()
        avg_accuracy = self.train_accuracy.compute()

        self.log("train/loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
        self.log("train/accuracy_epoch", avg_accuracy, prog_bar=True, on_epoch=True)

        # Reset metrics for the next epoch
        self.train_loss.reset()
        self.train_accuracy.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        logits = self.model(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)

        # Update metrics
        self.val_loss.update(loss)
        self.val_accuracy.update(preds, y)

        # Log metrics for this batch
        self.log("val/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(
            "val/accuracy_step",
            self.val_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

        return loss

    def on_validation_epoch_end(self):
        # Log average loss and metrics for the entire validation epoch
        avg_loss = self.val_loss.compute()
        avg_accuracy = self.val_accuracy.compute()

        self.log("val/loss_epoch", avg_loss, prog_bar=True, on_epoch=True)
        self.log("val/accuracy_epoch", avg_accuracy, prog_bar=True, on_epoch=True)

        # Reset metrics for the next epoch
        self.val_loss.reset()
        self.val_accuracy.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), self.learning_rate, weight_decay=self.weight_decay
        )


# Data
def load_data(ds, tr_train, tr_val):
    all_ids = [i for i in range(0, len(ds.observations))]
    all_labels = [x["label"] for x in ds.observations]

    train_ids, val_ids, test_ids = create_train_test_split(
        all_ids, all_labels, random_state=123, test_size=0.2, val_size=0.1
    )


    ds_train = ImageFolder.from_subset(ds, train_ids)
    ds_val = ImageFolder.from_subset(ds, val_ids)
    ds_test = ImageFolder.from_subset(ds, test_ids)

    dm = DataSetModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        classes=ds.classes,
        train_transform=tr_train,
        test_transform=tr_val,
        batch_size=64,
    )
    return dm

# Training Loop

# early stopping callback
early_stopping = EarlyStopping(
    monitor="val/accuracy_epoch",
    min_delta=0.01,
    patience=5,
    mode="max",
    verbose=True,
    strict=True,
)

def training_loop():

    L.seed_everything(123)
    logger = TensorBoardLogger(SCRIPT_DIR.joinpath("lightning_logs"), name="resnet18")

    max_epochs = 100
    max_steps = -1

    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        precision="32",
        max_epochs=max_epochs,
        max_steps=max_steps,
        fast_dev_run=False,
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[early_stopping],
        default_root_dir=SCRIPT_DIR.joinpath("lightning_logs"),
    )

    data_paths = [DATA_BASE_PATH / "photos", DATA_BASE_PATH / "renders"]
    #data_paths = SCRIPT_DIR / "data" / "data_set" / "three_classes"
    ds = ImageFolder(data_paths)

    num_classes = len(ds.classes)

    net, tr_train, tr_val = get_resnet18_model_and_transforms(num_classes)
    #print(torchinfo.summary(net, input_size=(1, 3, 64, 64)))

    dm = load_data(ds, tr_train, tr_val)
    l = [l["label"] for l in dm.ds_train.observations]
    plt.hist(l)
    plt.xticks(rotation=90)
    if len(l) > 15:
        plt.gca().set_xticklabels([])
    plt.show()

    model = Classifier(net, num_classes=num_classes, learning_rate=0.001, weight_decay=1e-4)

    trainer.fit(model, datamodule=dm)


def print_tensorboard_cmd():
    logdir = SCRIPT_DIR / "lightning_logs"
    print("Run in venv:")
    print(" ".join(['tensorboard', f'--logdir="{logdir}"', '--host=127.0.0.1', '--port=6006']))
    print(f"Log dir: {logdir}\nserver on: http://127.0.0.1:6006")

if __name__ == '__main__':
    training_loop()
    print_tensorboard_cmd()
