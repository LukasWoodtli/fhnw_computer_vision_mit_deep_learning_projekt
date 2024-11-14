import os
import zipfile
import pickle
from pathlib import Path

import torch
import torchmetrics
import torchvision
from PIL import Image
from prettytable import PrettyTable
from robust_downloader import download
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

import ssl


from torch import nn, log_softmax
from torch.utils.data import Subset, DataLoader
import torchvision.transforms.v2 as transforms
from torchvision.models import ResNet18_Weights, resnet18, resnet50
from torchvision.transforms.functional import to_pil_image
import lightning as L
import torch.nn.functional as F
from torchvision.transforms.v2 import Lambda

L.seed_everything(123)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")

try:
    import google.colab
    IN_COLAB = True
    accelerator="auto"
    assert torch.cuda.is_available()
except:
    IN_COLAB = False
    accelerator = "cpu"

print(f"In colab: {IN_COLAB}")


ssl._create_default_https_context = ssl._create_unverified_context

SCRIPT_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
DATA_DIR = SCRIPT_DIR / 'data'
SAVE_DIR = SCRIPT_DIR / 'save'
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)


url = "https://mostwiedzy.pl/en/open-research-data/lego-bricks-for-training-classification-network,202309140842198941751-0/download/"
filename = 'file 05 - dataset.zip'

def download_and_unpack(url, filename, data_dir):
    DATA_DOWNLOAD_DIR = os.path.join(data_dir, 'download')
    Path(DATA_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    DATA_ARCHIVE = os.path.join(DATA_DOWNLOAD_DIR, filename)
    if not os.path.exists(DATA_ARCHIVE):
        download(url, DATA_DOWNLOAD_DIR, filename)
    DATA_SET_DIR = Path(os.path.join(data_dir, 'data_set'))
    if not DATA_SET_DIR.exists():
        Path(DATA_SET_DIR).mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(DATA_ARCHIVE, 'r') as zipf:
            zipf.extractall(DATA_SET_DIR)
    return DATA_SET_DIR


data_set_dir = download_and_unpack(url, filename, DATA_DIR)


def verify_image(fn):
    """Confirm that `fn` can be opened (taken from fast.ai)"""
    try:
        im = Image.open(fn)
        im.draft(im.mode, (32,32))
        im.load()
        return True
    except: return False


number_of_classes = 447
photos_data_set = os.path.join(data_set_dir, 'photos')
DS_ALL_SAVE_FILE = SAVE_DIR / 'ds_all.pickle'
if DS_ALL_SAVE_FILE.exists():
    with open(DS_ALL_SAVE_FILE, 'rb') as pf:
        ds_all = pickle.load(pf)
else:
    # TODO
    data_set_transforms = transforms.Compose(
        [transforms.Grayscale(),
         transforms.Resize((64, 64)),
         transforms.ToTensor().to(torch.float32),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    ds_all = torchvision.datasets.ImageFolder(root=photos_data_set,
                                          transform = data_set_transforms,
                                          is_valid_file = verify_image,
                                          allow_empty = False)
    with open(DS_ALL_SAVE_FILE,'wb') as pf:
        pickle.dump(ds_all, pf)

all_ids = [i for i in range(0, len(ds_all.samples))]
all_labels = [i[1] for i in ds_all.samples]
all_classes = ds_all.classes
print(f"Data set size: {len(ds_all)}")
assert number_of_classes == len(all_classes)
print(f"Number of classes: {number_of_classes}")



# Split train, valid and test

train_ids, test_ids = train_test_split(
    all_ids,
    stratify=all_labels,
    test_size=0.2,
    random_state=123,
)

train_ids, val_ids = train_test_split(
    train_ids,
    stratify=[all_labels[i] for i in train_ids],
    test_size=0.1,
    random_state=123,
)

ds_train = Subset(ds_all, train_ids)
ds_val = Subset(ds_all, val_ids)
ds_test = Subset(ds_all, test_ids)


# Exploratory Data Analysis

ds_train_nb = len(ds_train)
ds_val_nb = len(ds_val)
ds_test_nb = len(ds_test)
total = ds_train_nb + ds_val_nb + ds_test_nb
assert total == len(ds_all)
table = PrettyTable()
table.field_names = ["Set", "Count", "%"]
table.add_row(["Train", ds_train_nb, f"{100.*ds_train_nb/total:.1f} %"])
table.add_row(["Validation", ds_val_nb, f"{100.*ds_val_nb/total:.1f} %"])
table.add_row(["Test", ds_test_nb, f"{100.*ds_test_nb/total:.1f} %"])
print(table)

# plot a grid
cols, rows = 3, 3
sample_ids = torch.randint(len(ds_train), size=(cols*rows,))
figure = plt.figure(figsize=(8, 8))
for i in range(0, cols * rows):
    sample_idx = sample_ids[i]
    img, label = ds_train[sample_idx]
    figure.add_subplot(rows, cols, i+1)
    plt.title(f"{ds_all.classes[label]} {img.shape}")
    plt.axis("off")
    img = to_pil_image(img, "RGB")
    plt.imshow(img, cmap="gray")
plt.show()


# plot a single image
def plot_image_with_label(id):
    img, label = ds_train[id]
    plt.title(f"{ds_all.classes[label]} {img.shape}")
    plt.axis("off")
    img = to_pil_image(img, "RGB")
    plt.imshow(img, cmap="gray")
    plt.show()

num = 133
plot_image_with_label(num)

img, _label = ds_train[num]
plt.hist(img.flatten(), bins=255)
plt.show()

# Dataloader
dataloader_train = DataLoader(ds_train, batch_size=64, shuffle=True)
dataloader_val = DataLoader(ds_val, batch_size=64, shuffle=True)
dataloader_test = DataLoader(ds_test, batch_size=64, shuffle=True)

# Model

class Classifier(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.lr = 1e-3
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy(
            task="multiclass",
            num_classes=number_of_classes,
        )  # Adjust task if you have more than two classes
        #self.train_loss = torchmetrics.MeanMetric()
        self.model = resnet50(pretrained=True)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.model.children())[-1].in_features
        # replace final layer for fine-tuning
        self.model.fc = nn.Linear(linear_size, number_of_classes)

        # only tune the fully-connected layers
        # for child in list(self.model.children())[:-1]:
        #     for param in child.parameters():
        #         param.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x, y = batch
        preds = self(x)

        loss = self.loss_fn(preds, y)
        acc = self.train_accuracy(preds, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        train_loss, train_accuracy = self._step(batch)
        self.log("train/acc_step", train_accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_step", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val/loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("test/loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test/acc", acc, on_step=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.lr)

fast_dev_run=False

trainer = L.Trainer(
    devices="auto",
    accelerator=accelerator,
    precision="32",
    fast_dev_run=fast_dev_run,
    max_steps=-1,
    max_epochs=2,
    enable_checkpointing=False,
    logger=False,
    default_root_dir=SCRIPT_DIR.joinpath("lightning_logs"),
)

def main():
    dataloader_train = DataLoader(ds_train, batch_size=32, shuffle=True, num_workers=1)

    model = Classifier()
    trainer.fit(model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    print(f"Metrics:  {trainer.logged_metrics}")

if __name__ == '__main__':
    main()
