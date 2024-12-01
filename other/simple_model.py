from pathlib import Path

from fastai.data.block import DataBlock, CategoryBlock
from fastai.data.transforms import get_image_files, RandomSplitter, parent_label
from fastai.metrics import error_rate
from fastai.vision.augment import Resize, aug_transforms
from fastai.vision.data import ImageBlock
from fastai.vision.learner import vision_learner
import torch
from torchvision import transforms
from torchvision.models import resnext50_32x4d

SCRIPT_DIR = Path(__file__).parent

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

data_set_transforms = transforms.Compose(
    [transforms.ToTensor().to(torch.float32),
     Resize(256),
     transforms.Normalize(mean=mean, std=std),
    ])

data_set_dir = SCRIPT_DIR / "data" / "data_set"
photos_data_set = data_set_dir / 'photos'
renders_data_set = data_set_dir / 'renders'

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=data_set_transforms,
    batch_tfms = aug_transforms
).dataloaders(photos_data_set)

learn = vision_learner(dls, resnext50_32x4d, metrics=error_rate)
learn.fine_tune(3)
