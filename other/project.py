from pathlib import Path

import torchmetrics
import torchvision.models as models
import lightning as L
from matplotlib import pyplot as plt
from ray.air import ScalingConfig, RunConfig, CheckpointConfig
from ray.train.huggingface.transformers import prepare_trainer
from ray.train.lightgbm import RayTrainReportCallback
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment
from ray.train.torch import TorchTrainer
from ray.tune import tune, choice, loguniform, Tuner, TuneConfig
from ray.tune.schedulers import ASHAScheduler

from utils import ImageFolder, create_train_test_split
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

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
def get_resnet18_model_and_transforms(num_classes, additional_train_transforms=None):
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net.fc = nn.Sequential(nn.Linear(512, num_classes))

    common_transforms = transforms.Compose(
        [
            transforms.v2.RGB(),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    if additional_train_transforms is not None:
        tr_train = transforms.Compose(
            [
                additional_train_transforms,
                common_transforms,
            ]
        )
    else:
        tr_train = common_transforms

    tr_val = common_transforms


    return net, tr_train, tr_val


def get_resnext101_model_and_transforms(num_classes, additional_train_transforms=None):
    net = models.resnext101_64x4d(weights=models.ResNeXt101_64X4D_Weights.DEFAULT)
    net.fc = nn.Sequential(nn.Linear(2048, num_classes))

    common_transforms = transforms.Compose([
            transforms.RandomResizedCrop(128),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    if additional_train_transforms is not None:
        tr_train = transforms.Compose(
            additional_train_transforms,
            common_transforms
        )
    else:
        tr_train = common_transforms

    tr_val = common_transforms

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
def load_data(ds, tr_train, tr_val, batch_size=64):
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
        batch_size=batch_size,
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

default_config = {
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
}

def training_loop(config=default_config):

    L.seed_everything(123)
    logger = TensorBoardLogger(SCRIPT_DIR.joinpath("lightning_logs"), name="resnext101")

    trainer = L.Trainer(
        devices="auto",
        accelerator="auto",
        precision="32",
        strategy=RayDDPStrategy(),
        fast_dev_run=False,
        enable_checkpointing=False,
        logger=logger,
        log_every_n_steps=1,
        callbacks=[early_stopping, RayTrainReportCallback()],
        plugins=[RayLightningEnvironment()],
        default_root_dir=SCRIPT_DIR.joinpath("lightning_logs"),
    )
    trainer = prepare_trainer(trainer)

    data_paths = [DATA_BASE_PATH / "photos", DATA_BASE_PATH / "renders"]
    #data_paths = SCRIPT_DIR / "data" / "data_set" / "three_classes"
    ds = ImageFolder(data_paths)

    num_classes = len(ds.classes)

    net, tr_train, tr_val = get_resnext101_model_and_transforms(num_classes)
    #print(torchinfo.summary(net, input_size=(1, 3, 64, 64)))

    dm = load_data(ds, tr_train, tr_val, )
    # l = [l["label"] for l in dm.ds_train.observations]
    # plt.hist(l)
    # plt.xticks(rotation=90)
    # if len(l) > 15:
    #     plt.gca().set_xticklabels([])
    #plt.show()

    model = Classifier(net, num_classes=num_classes,
                       learning_rate=config["learning_rate"],
                       weight_decay=config["weight_decay"])

    trainer.fit(model, datamodule=dm)


complex_transforms_0 = transforms.Compose(
    [
        transforms.RandomRotation(12),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    ])

complex_transforms_1 = transforms.Compose(
    [
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2)],
            p=0.8,
        ),
        transforms.RandomGrayscale(p=0.2),
    ])

search_space = {
    "additional_transforms": choice([None, complex_transforms_0, complex_transforms_1]),
    "learning_rate": loguniform(1e-4, 1e-1),
    "weight_decay": loguniform(1e-4, 1e-2),
    "batch_size": choice([32, 64]),
}


def tune():
    max_epochs = 20
    num_samples = 10

    scheduler = ASHAScheduler(max_t=max_epochs, grace_period=1, reduction_factor=2)

    scaling_config = ScalingConfig(
        num_workers=3, use_gpu=True, resources_per_worker={"CPU": 1, "GPU": 2}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="ptl/val_accuracy",
            checkpoint_score_order="max",
        ),
    )

    ray_trainer = TorchTrainer(
        training_loop,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    tuner = Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=TuneConfig(
            metric="ptl/val_accuracy",
            mode="max",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()

def print_tensorboard_cmd():
    logdir = SCRIPT_DIR / "lightning_logs"
    print("Run in venv:")
    print(" ".join(['tensorboard', f'--logdir="{logdir}"', '--host=127.0.0.1', '--port=6006']))
    print(f"Log dir: {logdir}\nserver on: http://127.0.0.1:6006")

if __name__ == '__main__':
    results = tune()
    print(results)


    print_tensorboard_cmd()
