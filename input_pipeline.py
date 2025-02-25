# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet input pipeline."""

import numpy as np
import os
import random
import jax
import torch
from torch.utils.data import DataLoader, default_collate
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from absl import logging
from functools import partial
from utils.logging_utils import log_for_0
from utils.aug_util import ContrastiveLearningViewGenerator, get_simclr_pipeline_transform


IMAGE_SIZE = 224
CROP_PADDING = 32
MEAN_RGB = [0.485, 0.456, 0.406]
STDDEV_RGB = [0.229, 0.224, 0.225]


def prepare_batch_data(batch, batch_size=None):
    """Reformat a input batch from PyTorch Dataloader.

    Args:
      batch = (image, label)
        image: shape (host_batch_size, 3, height, width)
        label: shape (host_batch_size)
      batch_size = expected batch_size of this node, for eval's drop_last=False only
    """
    image, label = batch
    # print(f"len: {len(image)}") # 2
    # print(f"image0 shape: {image[0].shape}") # (bs, 3, 224, 224)
    # print(f"image1 shape: {image[1].shape}")
    # print(f"label shape: {label.shape}") # (bs,)
    # exit("东灵")
    if isinstance(image, list): # training
        assert len(image) == 2 # [[1, 2, 3, 4], [1', 2', 3', 4']]
        image = torch.cat(image, axis=0)
        # image = torch.stack(image, axis=0).transpose(0, 1).reshape(-1, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert batch_size is None, NotImplementedError

        label = torch.cat([label, label], axis=0)

        # print(f"image shape: {image.shape}")
        # print(f"label shape: {label.shape}")

        # exit("东灵")

    # pad the batch if smaller than batch_size
    if batch_size is not None and batch_size > image.shape[0]:
        image = torch.cat(
            [
                image,
                torch.zeros(
                    (batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype
                ),
            ],
            axis=0,
        )
        label = torch.cat(
            [label, -torch.ones((batch_size - label.shape[0],), dtype=label.dtype)],
            axis=0,
        )

    # reshape (host_batch_size, 3, height, width) to
    # (local_devices, device_batch_size, height, width, 3)
    local_device_count = jax.local_device_count()
    image = image.permute(0, 2, 3, 1)
    image = image.reshape((local_device_count, -1) + image.shape[1:])
    label = label.reshape(local_device_count, -1)

    image = image.numpy()
    label = label.numpy()

    return_dict = {
        "image": image,
        "label": label,
    }

    return return_dict

def prepare_linear_eval_batch_data(batch, p_representation, batch_size=None):
    """Reformat a input batch from PyTorch Dataloader.

    Args:
      batch = (image, label)
        image: shape (host_batch_size, 3, height, width)
        label: shape (host_batch_size)
      batch_size = expected batch_size of this node, for eval's drop_last=False only
    """
    image, label = batch

    # pad the batch if smaller than batch_size
    if batch_size is not None and batch_size > image.shape[0]:
        image = torch.cat(
            [
                image,
                torch.zeros(
                    (batch_size - image.shape[0],) + image.shape[1:], dtype=image.dtype
                ),
            ],
            axis=0,
        )
        label = torch.cat(
            [label, -torch.ones((batch_size - label.shape[0],), dtype=label.dtype)],
            axis=0,
        )

    # reshape (host_batch_size, 3, height, width) to
    # (local_devices, device_batch_size, height, width, 3)
    local_device_count = jax.local_device_count()
    image = image.permute(0, 2, 3, 1)
    image = image.reshape((local_device_count, -1) + image.shape[1:])
    label = label.reshape(local_device_count, -1)

    image = image.numpy()
    label = label.numpy()

    representation = p_representation(x=image)

    return_dict = {
        "representation": representation,
        "label": label,
    }

    return return_dict


def worker_init_fn(worker_id, rank):
    seed = worker_id + rank * 1000
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


from torchvision.datasets.folder import pil_loader


def loader(path: str):
    return pil_loader(path)


def create_split(
    dataset_cfg,
    batch_size,
    split,
):
    """Creates a split from the ImageNet dataset using Torchvision Datasets.

    Args:
      dataset_cfg: Configurations for the dataset.
      batch_size: Batch size for the dataloader.
      split: 'train' or 'val'.
    Returns:
      it: A PyTorch Dataloader.
      steps_per_epoch: Number of steps to loop through the DataLoader.
    """
    rank = jax.process_index()
    if split == "train":
        ds = datasets.ImageFolder(
            os.path.join(dataset_cfg.root, split),
            # transform=transforms.Compose(
            #     [
            #         transforms.RandomResizedCrop(IMAGE_SIZE, interpolation=3),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.ToTensor(),
            #         # transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
            #     ]
            # ),
            transform=ContrastiveLearningViewGenerator(
                base_transform=get_simclr_pipeline_transform(IMAGE_SIZE),
                n_views=2,
            ),
            loader=loader,
        )
        log_for_0(ds)
        sampler = DistributedSampler(
            ds,
            num_replicas=jax.process_count(),
            rank=rank,
            shuffle=True,
        )
        it = DataLoader(
            ds,
            batch_size=batch_size,
            drop_last=True,
            worker_init_fn=partial(worker_init_fn, rank=rank),
            sampler=sampler,
            num_workers=dataset_cfg.num_workers,
            prefetch_factor=(
                dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
            ),
            pin_memory=dataset_cfg.pin_memory,
            persistent_workers=True if dataset_cfg.num_workers > 0 else False,
        )
        steps_per_epoch = len(it)
    elif split == "linear_eval": # NOTE: here, for representation evaluation, we train the linear head on the train set
        ds = datasets.ImageFolder(
            os.path.join(dataset_cfg.root, "train"),
            transform=transforms.Compose(
                [
                    # transforms.Resize(IMAGE_SIZE + CROP_PADDING, interpolation=3),
                    # transforms.CenterCrop(IMAGE_SIZE),
                    transforms.RandomResizedCrop(IMAGE_SIZE, interpolation=3),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
                ]
            ),
            loader=loader,
        )
        log_for_0(ds)
        sampler = DistributedSampler(
            ds,
            num_replicas=jax.process_count(),
            rank=rank,
            shuffle=True, 
        )
        it = DataLoader(
            ds,
            batch_size=batch_size,
            drop_last=True, 
            worker_init_fn=partial(worker_init_fn, rank=rank),
            sampler=sampler,
            num_workers=dataset_cfg.num_workers,
            prefetch_factor=(
                dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
            ),
            pin_memory=dataset_cfg.pin_memory,
            persistent_workers=True if dataset_cfg.num_workers > 0 else False,
        )
        steps_per_epoch = len(it)
    elif split == "val": # This is for eval acc
        ds = datasets.ImageFolder(
            os.path.join(dataset_cfg.root, "val"),
            transform=transforms.Compose(
                [
                    transforms.Resize(IMAGE_SIZE + CROP_PADDING, interpolation=3),
                    transforms.CenterCrop(IMAGE_SIZE),
                    # transforms.RandomResizedCrop(IMAGE_SIZE, interpolation=3),
                    # transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=MEAN_RGB, std=STDDEV_RGB),
                ]
            ),
            loader=loader,
        )
        log_for_0(ds)
        """
    The val has 50000 images. We want to eval exactly 50000 images. When the
    batch is too big (>16), this number is not divisible by the batch size. We
    set drop_last=False and we will have a tailing batch smaller than the batch
    size, which requires modifying some eval code.
    """
        sampler = DistributedSampler(
            ds,
            num_replicas=jax.process_count(),
            rank=rank,
            shuffle=False,  # don't shuffle for val
        )
        it = DataLoader(
            ds,
            batch_size=batch_size,
            drop_last=False,  # don't drop for val
            worker_init_fn=partial(worker_init_fn, rank=rank),
            sampler=sampler,
            num_workers=dataset_cfg.num_workers,
            prefetch_factor=(
                dataset_cfg.prefetch_factor if dataset_cfg.num_workers > 0 else None
            ),
            pin_memory=dataset_cfg.pin_memory,
            persistent_workers=True if dataset_cfg.num_workers > 0 else False,
        )
        steps_per_epoch = len(it)
    else:
        raise NotImplementedError

    logging.info(f"Rank {rank}: dataset is loaded.")
    return it, steps_per_epoch
