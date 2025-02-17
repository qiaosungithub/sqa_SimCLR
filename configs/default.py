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

# Copyright 2021 The Flax Authors.
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
"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # Model
    config.model = model = ml_collections.ConfigDict()
    model.half_precision = True

    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()
    dataset.name = "imagenet"
    dataset.root = "/kmh-nfs-us-mount/data/imagenet"
    dataset.num_workers = 4
    dataset.prefetch_factor = 2
    dataset.pin_memory = False
    dataset.cache = True
    dataset.image_size = 224
    dataset.num_classes = 1000

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.learning_rate = 0.1
    training.warmup_epochs = 5
    training.momentum = 0.9
    training.weight_decay = 1e-4
    training.shuffle_buffer_size = 16 * 1024
    # training.prefetch = 10 # Don't know what exactly it is
    training.num_epochs = 100
    training.wandb = True
    training.log_per_step = 100
    training.log_per_epoch = -1
    training.eval_per_epoch = 1
    training.checkpoint_per_epoch = 20
    training.checkpoint_max_keep = 2
    training.steps_per_eval = -1
    training.seed = 3407  # init random seed

    # eval
    config.evalu = evalu = ml_collections

    # wandb
    config.wandb = True

    return config


def metrics():
    return [
        "train_loss",
        "eval_loss",
        "train_accuracy",
        "eval_accuracy",
        "steps_per_second",
        "train_learning_rate",
    ]
