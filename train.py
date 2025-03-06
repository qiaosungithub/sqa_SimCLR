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
from typing import Any
from functools import partial

# from absl import logging
from flax import jax_utils
from flax.training import checkpoints, common_utils, train_state
import jax, ml_collections, optax, wandb, re, os
from jax import lax, random
import jax.numpy as jnp
import flax.linen as nn

import input_pipeline
from input_pipeline import prepare_batch_data, prepare_linear_eval_batch_data
# import models.models_resnet as models_resnet
from models.models_simclr import SimCLR

from utils.info_util import print_params
from utils.logging_utils import log_for_0, GoodLogger
from utils.metric_utils import Timer, MyMetrics


def create_model(*, model_cls, half_precision, num_classes, **kwargs):
    """
    Create a model using the given model class.
    """
    raise NotImplementedError
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == "tpu":
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def initialized(key, shape, model):
    """
    Initialize the model, and return the model parameters.
    """
    # input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0("Initializing params...")
    variables = init({"params": key}, jnp.ones(shape, model.dtype))
    if "batch_stats" not in variables:
        variables["batch_stats"] = {}
    log_for_0("Initializing params done.")
    return variables["params"], variables["batch_stats"]


def create_learning_rate_fn(
    training_config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int,
):
    """
    Create learning rate schedule.
    """
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=base_learning_rate,
        transition_steps=training_config.warmup_epochs * steps_per_epoch,
    )
    cosine_epochs = max(training_config.num_epochs - training_config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[training_config.warmup_epochs * steps_per_epoch],
    )
    return schedule_fn


class TrainState(train_state.TrainState):
    batch_stats: Any

def get_no_weight_decay_dict(params):
    def modify_value_based_on_key(obj):
        if not isinstance(obj, dict):
            return obj
        for k,v in obj.items():
            if not isinstance(v,dict):
                if k in {'bias','scale'}: # scale is for batch norm
                    obj[k] = False
                else:
                    obj[k] = True
        return obj
    def is_leaf(obj):
        if not isinstance(obj, dict):
            return True
        modify_value_based_on_key(obj)
        b = isinstance(obj, dict) and all([not isinstance(v, dict) for v in obj.values()])
        return b
    u = jax.tree_util.tree_map(lambda x:False,params)
    modified_tree = jax.tree_util.tree_map(partial(modify_value_based_on_key), u, is_leaf=is_leaf)
    return modified_tree

def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
    """
    Create initial training state.
    """
    platform = jax.local_devices()[0].platform
    if config.model.half_precision and platform == "gpu":
        raise NotImplementedError("We consider TPU only.")

    s = config.dataset.image_size
    params, batch_stats = initialized(rng, (1, s, s, 3), model)

    log_for_0(params, logging_fn=print_params)

    if config.training.optimizer == "sgd":
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.training.momentum,
            nesterov=True,
        )
    elif config.training.optimizer == "LARS":
        no_weight_decay_mask = get_no_weight_decay_dict(params)
        tx = optax.lars(
            learning_rate=learning_rate_fn,
            weight_decay=config.training.weight_decay,
            weight_decay_mask=no_weight_decay_mask,
        )
    else: raise NotImplementedError
    state = TrainState.create(
        # apply_fn=partial(model.apply, method=model.forward),
        apply_fn=model.apply,
        params=params,
        tx=tx,
        batch_stats=batch_stats,
    )
    return state


def compute_metrics(logits, labels, num_classes):
    """
    Utils function to compute metrics, used in train_step and eval_step.
    """
    # compute per-sample loss
    one_hot_labels = common_utils.onehot(labels, num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    loss = xentropy  # (local_batch_size,)

    accuracy = jnp.argmax(logits, -1) == labels  # (local_batch_size,)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    metrics = lax.all_gather(metrics, axis_name="batch")
    metrics = jax.tree_map(lambda x: x.flatten(), metrics)  # (batch_size,)
    return metrics


def cross_entropy_loss(logits, labels, num_classes):
    """
    Utils function to compute training loss.
    """
    one_hot_labels = common_utils.onehot(labels, num_classes)
    xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
    return jnp.mean(xentropy)

def NTXent(features, temperature=0.1):
    """
    Tricky here: labels stand for the index of image, the same label corresponds to the same image
    e.g. bs=2048
    Each machine (process) has 512 different images
    i.e. (1, ..., 512, 1', ..., 512')
    and after all gather, becomes (1, ..., 512, 1', ..., 512', 513, ..., 1024, 513', ..., 1024', ...)
    """
    assert features.ndim == 3
    features = features.reshape(-1, features.shape[-1])
    B = features.shape[0] // 2
    # # generate mask
    # labels = jnp.concatenate([jnp.arange(B) for _ in range(2)], axis=0)
    n_process = jax.process_count()
    B_ = B // n_process # batch size per process
    labels_per_process = jnp.concatenate([jnp.arange(B_) for _ in range(2)], axis=0)
    labels_all = labels_per_process.reshape(1, 2*B_).repeat(n_process, axis=0)
    labels_all = labels_all + jnp.arange(n_process).reshape(-1, 1) * 2 * B_
    labels_ = labels_all.reshape(-1)
    labels = (labels_[:, None] == labels_[None, :]).astype(jnp.float32)
    # mask = jnp.eye(2 * B, dtype=jnp.float32)
    # assert False, f"labels shape: {labels.shape}, mask shape: {mask.shape}"
    # mask_indices = jnp.where(~mask)
    # mask = jax.device_get(mask)  # 将 mask 转换为 NumPy 数组
    # labels = labels[~mask].reshape(2 * B, 2 * B - 1)
    labels = labels - jnp.eye(2 * B, dtype=jnp.float32)
    # compute similarity matrix
    # features = features / jnp.linalg.norm(features, axis=-1, keepdims=True)
    similarity_matrix = jnp.dot(features, features.T)
    assert similarity_matrix.shape == (2 * B, 2 * B)
    # similarity_matrix = similarity_matrix[~mask].reshape(2 * B, 2 * B - 1)
    # similarity_matrix = similarity_matrix + jnp.eye(2 * B, dtype=jnp.float32) * (-jnp.inf) # remove diagonal similarity
    similarity_matrix = jnp.fill_diagonal(similarity_matrix, -jnp.inf, inplace=False) # remove diagonal similarity
    # compute logits
    # labels = labels.astype(jnp.bool)
    # labels = jax.device_get(labels)
    # positives = similarity_matrix[labels].reshape(2 * B, 1)
    # negatives = similarity_matrix[~labels].reshape(2 * B, 2 * B-2)

    # print(f"B: {B}, positives shape: {positives.shape}, negatives shape: {negatives.shape}")
    # exit("路明")

    # logits = jnp.concatenate([positives, negatives], axis=1)
    logits = similarity_matrix / temperature
    # compute loss
    # legacy
    # prob = jax.nn.softmax(logits, axis=1)
    # # prob = prob[:, 0]
    # prob = jnp.sum(prob * labels, axis=1)
    # loss = jnp.mean(-jnp.log(prob))

    # optimized version
    logits0 = jnp.fill_diagonal(logits, 0, inplace=False)
    loss = - jnp.sum(logits0 * labels, axis=1) + nn.logsumexp(logits, axis=1)
    loss = jnp.mean(loss)
    # compute acc
    pred = jnp.argmax(logits, axis=1)
    correct_label = jnp.argmax(labels, axis=1)
    acc = jnp.mean(pred == correct_label)
    # acc = jnp.mean(pred == 0)

    # debug dict
    d = {}
    # d["features"] = features
    # d["similarity_matrix"] = similarity_matrix
    # d["labels"] = labels
    # d["labels_"] = labels_
    # # d["correct_label"] = correct_label
    # d["logits"] = logits
    # d["logsumexp"] = nn.logsumexp(logits, axis=1)
    # d["another"] = jnp.sum(logits * labels, axis=1)
    # d["loss"] = loss
    return loss, acc, d

def train_step(state, batch, rng_init, learning_rate_fn, weight_decay, config):
    """
    Perform a single training step. This function will be pmap, so we can't print inside it.
    """
    # ResNet has no dropout; but maintain rng_dropout for future usage
    rng_step = random.fold_in(rng_init, state.step)
    rng_device = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))
    rng_dropout, _ = random.split(rng_device)

    images = batch["image"]

    def loss_fn(params):
        features, new_model_state = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            images,
            mutable=["batch_stats"],
            rngs=dict(dropout=rng_dropout),
        )
        # gather all features
        features = features / jnp.linalg.norm(features, axis=-1, keepdims=True) # normalize features
        all_features = lax.all_gather(features, axis_name="batch")
        # all_features = lax.stop_gradient(all_features)
        # rank = lax.axis_index(axis_name="batch")
        # all_features = all_features.at[rank].set(features) # keep the gradient of features on this device
        # assert False, f"features shape: {features.shape}, images shape: {images.shape}" # feature: (8, 2b_2, c), images: (2b_2, 224, 224, 3)
        loss, acc, d = NTXent(all_features, temperature=config.training.temperature)
        # loss = cross_entropy_loss(logits, batch["label"], num_classes)
        # weight_penalty_params = jax.tree_util.tree_leaves(params)
        # weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
        # weight_penalty = weight_decay * 0.5 * weight_l2
        # loss = loss + weight_penalty
        # loss = 0.1
        # acc = 0.1
        # d = {"rank": rank}
        return loss, (new_model_state, acc, d)

    # compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name="batch")
    new_model_state, acc, d = aux[1]

    # d["grad"] = grads
    # apply gradients
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )

    # compute metrics
    # metrics = compute_metrics(logits, batch["label"], num_classes)
    metrics = {"loss": aux[0], "contrastive_acc": acc}
    metrics["lr"] = learning_rate_fn(state.step)
    return new_state, metrics, d

class LinearHead(nn.Module):
    num_classes: int=1000
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros_init())(x)
        return x

def init_eval(state, config, hidden_dim, learning_rate_fn, model):
    """
    Initialize evaluation step.
    """
    # representation function compiled
    p_representation = jax.pmap(
        partial(
            model.apply, 
            variables={"params":state.params, "batch_stats":state.batch_stats},
            method=model.forward,
            mutable=False,
        ),
        axis_name="batch"
    )
    # initialize head
    head = LinearHead(num_classes=config.dataset.num_classes)
    params, _ = initialized(random.PRNGKey(0), (1, hidden_dim), head)
    log_for_0(params, logging_fn=print_params)
    weight_decay_mask = get_no_weight_decay_dict(params)
    # create train state
    tx = optax.lars(
        learning_rate=learning_rate_fn,
        # momentum=0.9,
        # nesterov=True,
        weight_decay=config.training.weight_decay,
        weight_decay_mask=weight_decay_mask,
    )
    head_state = TrainState.create(
        apply_fn=head.apply,
        params=params,
        tx=tx,
        batch_stats={},
    )
    return p_representation, head_state

def linear_eval_step(head_state, batch, rng_init, num_classes):
    """
    train one step for the linear head, with representation in the batch
    """
    # ResNet has no dropout; but maintain rng_dropout for future usage
    rng_step = random.fold_in(rng_init, head_state.step)
    rng_device = random.fold_in(rng_step, lax.axis_index(axis_name="batch"))
    rng_dropout, _ = random.split(rng_device)

    representation = batch["representation"]
    labels = batch["label"]

    def loss_fn(params):
        logits = head_state.apply_fn(
            {"params": params}, 
            representation, 
            mutable=False,
            rngs=dict(dropout=rng_dropout),
        )
        loss = cross_entropy_loss(logits, labels, num_classes)
        return loss, logits

    # compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(head_state.params)
    grads = lax.pmean(grads, axis_name="batch")
    loss = aux[0]
    logits = aux[1]

    # apply gradients
    new_head_state = head_state.apply_gradients(grads=grads)

    # compute metrics
    metrics = compute_metrics(logits, labels, num_classes)
    return new_head_state, metrics

def eval_step(state, batch, num_classes):
    """
    Perform a single evaluation step. This function will be pmap, so we can't print inside it.
    """
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, batch["representation"], mutable=False)
    metrics = compute_metrics(logits, batch["label"], num_classes)
    return metrics


def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
    state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
    step = int(state.step)
    log_for_0("Saving checkpoint step %d.", step)
    checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=2)


@partial(jax.pmap, axis_name="x")
def cross_replica_mean(x):
    """
    Compute an average of a variable across workers.
    """
    return lax.pmean(x, "x")


def sync_batch_stats(state):
    """Sync the batch statistics across replicas."""
    # Each device has its own version of the running average batch statistics and
    # we sync them before evaluation.
    # if there is not batch_stats, we don't sync
    # return state
    # try:
    #     new_batch_stats = cross_replica_mean(state.batch_stats)
    # except:
    #     new_batch_stats = state.batch_stats
    new_batch_stats = cross_replica_mean(state.batch_stats)
    return state.replace(batch_stats=new_batch_stats)


def train_and_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    # # test
    # print("process count: ", jax.process_count(), flush=True) # 4
    # print("process index: ", jax.process_index(), flush=True)
    # exit("邓")
    # print(f"{jax.device_count()}", flush=True)
    ######################################################################
    #                       Initialize training                          #
    ######################################################################
    training_config = config.training
    if jax.process_index() == 0 and training_config.wandb:
        wandb.init(project="sqa_simclr", dir=workdir)
        wandb.config.update(config.to_dict())
        ka = re.search(r"kmh-tpuvm-v[234]-(\d+)(-preemptible)?-(\d+)", workdir).group()
        wandb.config.update({"ka": ka})

    rank = jax.process_index()

    logger = GoodLogger(use_wandb=training_config.wandb)

    rng = random.key(training_config.seed)

    global_batch_size = training_config.batch_size
    log_for_0("config.batch_size: {}".format(global_batch_size))

    if global_batch_size % jax.process_count() > 0:
        raise ValueError("Batch size must be divisible by the number of processes")
    local_batch_size = global_batch_size // jax.process_count()
    log_for_0("local_batch_size: {}".format(local_batch_size))
    log_for_0("jax.local_device_count: {}".format(jax.local_device_count()))

    if local_batch_size % jax.local_device_count() > 0:
        raise ValueError(
            "Local batch size must be divisible by the number of local devices"
        )

    ######################################################################
    #                           Create Dataloaders                       #
    ######################################################################

    train_loader, steps_per_epoch = input_pipeline.create_split(
        config.dataset,
        local_batch_size,
        split="train",
    )
    # linear_eval_loader, steps_per_linear_eval = input_pipeline.create_split(
    #     config.dataset,
    #     local_batch_size,
    #     split="linear_eval",
    # )
    # eval_loader, steps_per_eval = input_pipeline.create_split(
    #     config.dataset,
    #     local_batch_size,
    #     split="val",
    # )
    log_for_0("steps_per_epoch: {}".format(steps_per_epoch))
    # log_for_0("steps_per_linear_eval: {}".format(steps_per_linear_eval))
    # log_for_0("steps_per_eval: {}".format(steps_per_eval))

    # if training_config.steps_per_eval != -1:
    #     steps_per_eval = training_config.steps_per_eval

    ######################################################################
    #                       Create Train State                           #
    ######################################################################

    lr_scaling = config.training.lr_scaling
    lr = config.training.learning_rate
    if lr_scaling == "linear":
        base_learning_rate = lr * global_batch_size / 256.0
    elif lr_scaling == "sqrt":
        base_learning_rate = lr * jnp.sqrt(global_batch_size)

    # model_cls = SimCLR
    # model = create_model(
    #     model_cls=model_cls,
    #     half_precision=config.model.half_precision,
    #     num_classes=config.dataset.num_classes,
    # )
    net_type = config.model.name
    # get corresponding hidden_dim
    if net_type == "ResNet50": hidden_dim = 2048
    elif net_type == "_ResNet1": hidden_dim = 64
    else: raise NotImplementedError(f"model {net_type} not implemented")
    model = SimCLR(net_type=net_type, hidden_dim=hidden_dim)

    learning_rate_fn = create_learning_rate_fn(
        training_config, base_learning_rate, steps_per_epoch
    )

    state = create_train_state(
        rng, config, model, config.dataset.image_size, learning_rate_fn
    )
    # state = restore_checkpoint(state, workdir) # restore checkpoint
    # step_offset = int(state.step)
    # epoch_offset = step_offset // steps_per_epoch
    # assert (
    #     epoch_offset * steps_per_epoch == step_offset
    # ), "Your checkpoint step {} is not aligned with steps_per_epoch {}".format(
    #     step_offset, steps_per_epoch
    # )
    epoch_offset = 0
    state = jax_utils.replicate(state)

    ######################################################################
    #                     Prepare for Training Loop                      #
    ######################################################################
    # B = config.training.batch_size
    # mask = jnp.eye(2 * B, dtype=jnp.bool)

    # labels_for_compute = jnp.concatenate([jnp.arange(B) for _ in range(2)], axis=0)
    # labels_for_compute = (labels_for_compute[:, None] == labels_for_compute[None, :]).astype(jnp.float32)
    # labels_for_compute = labels_for_compute[~mask].reshape(2 * B, 2 * B - 1)
    # labels_for_compute = labels_for_compute.astype(jnp.bool)

    p_train_step = jax.pmap(
        partial(
            train_step,
            rng_init=rng,
            learning_rate_fn=learning_rate_fn,
            # num_classes=config.dataset.num_classes,
            weight_decay=training_config.weight_decay,
            config=config,
            # mask=mask,
            # labels_for_compute=labels_for_compute,
        ),
        axis_name="batch",
    )
    # p_linear_eval_step = jax.pmap(
    #     partial(
    #         linear_eval_step, 
    #         rng_init=rng,
    #         num_classes=config.dataset.num_classes
    #     ),
    #     axis_name="batch",
    # )
    # p_eval_step = jax.pmap(
    #     partial(eval_step, num_classes=config.dataset.num_classes),
    #     axis_name="batch",
    # )

    log_for_0("Initial compilation, this might take some minutes...")

    ######################################################################
    #                           Training Loop                            #
    ######################################################################
    timer = Timer()
    for epoch in range(epoch_offset, training_config.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0("epoch {}...".format(epoch))

        # training
        train_metrics = MyMetrics(reduction="last")
        for n_batch, batch in enumerate(train_loader):
            batch = prepare_batch_data(batch)

            # # sanity check

            # # Uncomment this
            # def tang(i): 
            #     i = lax.all_gather(i, axis_name="batch")
            #     return i
            # p_tang = jax.pmap(tang, axis_name="batch")
            # images = batch["image"]
            # # print(f"images.shape1: {images.shape}") # if run on 32 TPUs, e.g. bs = 352, then images.shape = (8, 22, 224, 224, 3), and each machine has one such images
            # images = p_tang(images)
            # images = images[0]
            # print(f"images.shape2: {images.shape}")
            # # exit("邓东灵")
            # images = images.reshape(-1, 224, 224, 3)
            # B = images.shape[0] // 2
            # # exit("邓东灵")
            # import matplotlib.pyplot as plt
            # import os
            # path = f"/kmh-nfs-ssd-eu-mount/staging/sqa/debug-kmh-tpuvm-v3-32-1/total/"
            # if os.path.exists(path) == False: os.makedirs(path)
            # for i in range(2*B):
            #     img = images[i]
            #     plt.imsave(path+f"{i}.png", img)
            # exit("邓东灵")
            # # END

            # # print("batch['image'].shape:", batch['image'].shape)
            # # assert False

            # # # here is code for us to visualize the images
            # import matplotlib.pyplot as plt
            # import numpy as np
            # import os
            # images = batch["image"]
            # print(f"images.shape: {images.shape}", flush=True)
            # print(f'image max: {jnp.max(images)}, min: {jnp.min(images)}') # here, [0, 1] TODO: whether to transform it into [-1, 1]

            # # from input_pipeline import MEAN_RGB, STDDEV_RGB

            # # save batch["image"] to ./images/{epoch}/i.png
            # rank = jax.process_index()

            # # if os.path.exists(f"/kmh-nfs-us-mount/staging/sqa/images/{n_batch}/{rank}") == False:
            # #   os.makedirs(f"/kmh-nfs-us-mount/staging/sqa/images/{n_batch}/{rank}")
            # path = f"/kmh-nfs-ssd-eu-mount/code/qiao/work/dataset_images/{n_batch}/{rank}"
            # if os.path.exists(path) == False:
            #   os.makedirs(path)
            # for i in range(images[0].shape[0]):
            #     # print the max and min of the image
            #     # print(f"max: {np.max(images[0][i])}, min: {np.min(images[0][i])}")
            #     # img_test = images[0][:100]
            #     # save_img(img_test, f"/kmh-nfs-ssd-eu-mount/code/qiao/flow-matching/sqa_flow-matching/dataset_images/{n_batch}/{rank}", im_name=f"{i}.png", grid=(10, 10))
            #     # break
            #     # use the max and min to normalize the image to [0, 1]
            #     img = images[0][i]
            #     # img = img * (jnp.array(STDDEV_RGB)/255.).reshape(1,1,3) + (jnp.array(MEAN_RGB)/255.).reshape(1,1,3)
            #     # img = (img + 1) / 2
            #     # print(f"max: {np.max(img)}, min: {np.min(img)}")
            #     img = jnp.clip(img, 0, 1)
            #     # img = (img - np.min(img)) / (np.max(img) - np.min(img))
            #     # img = img.squeeze(-1)
            #     plt.imsave(path+f"/{i}.png", img) # if MNIST, add cmap='gray'
            #     if i>20: break

            # print(f"saving images for n_batch {n_batch}, done.")
            # if n_batch > 0:
            #   exit(114514)
            # continue

            state, metrics, d = p_train_step(state, batch)
            train_metrics.update(metrics)

            # # debug
            # for k, v in d.items():
            #     print(f"{k}: {v.shape}")
            #     v = v[0]
            #     print(f"{k}: {v}", flush=True)
            #     if k == "labels_":
            #         print(v[0], v[88])
            #     # print(f"{jnp.any(jnp.isnan(v))}")
            #     # d[k] = v[rank]
            # exit("邓东灵")
            # grad = d["grad"] # a dict
            # def show(d, rank): 
            #     for k, v in d.items():
            #         if isinstance(v, dict):
            #             print(f"{k}:\n")
            #             show(v, rank)
            #             print("\n")
            #         else:
            #             print(f"{k}: {v.shape}, magnitude: {jnp.linalg.norm(v[rank])}")
            # show(grad, 0)
            # show(grad, 1)

            
            # exit("丁")
            # import os
            # path = f"/kmh-nfs-ssd-eu-mount/staging/sqa/debug-kmh-tpuvm-v3-32-1/{rank}/"
            # if os.path.exists(path) == False:
            #     os.makedirs(path)
            # with open(path+"debug.txt", "w") as f:
            #     f.write(f"correct_label: {d['correct_label']}\n\n\n")
            #     f.write(f"features: {d['features']}\n\n\n")
            #     f.write(f"similarity_matrix: {d['similarity_matrix']}\n\n\n")
            #     f.write(f"loss: {metrics['loss']}\n\n\n")
            #     # f.write(f"grad: {d['grad']}\n\n\n")
            # exit("邓东灵")

            if epoch == epoch_offset and n_batch == 0:
                log_for_0("Initial compilation completed. Reset timer.")
                timer.reset()

            step = epoch * steps_per_epoch + n_batch
            ep = epoch + n_batch / steps_per_epoch
            if training_config.get("log_per_step"):
                if (step + 1) % training_config.log_per_step == 0:
                    # compute and log metrics
                    summary = train_metrics.compute_and_reset()
                    summary = {f"train_{k}": v for k, v in summary.items()}
                    summary["steps_per_second"] = (
                        training_config.log_per_step / timer.elapse_with_reset()
                    )
                    summary.update({"ep": ep, "step": step})
                    logger.log(step + 1, summary)

            # # print(f"state.batch_stats: {state.batch_stats}")
            # d = state.batch_stats
            # def show_dict(d):
            #     for k, v in d.items():
            #         if isinstance(v, dict):
            #             print(f"{k}:")
            #             show_dict(v)
            #         else:
            #             print(f"{k}: {v.shape}")
            # show_dict(d)
            # exit("邓东灵")
        # # linear evaluation
        # if (epoch + 1) % training_config.eval_per_epoch == 0:
        #     log_for_0("Eval epoch {}...".format(epoch))
        #     # sync batch statistics across replicas
        #     state = sync_batch_stats(state)
        #     eval_metrics = MyMetrics(reduction="avg")

        #     for n_eval_batch, eval_batch in enumerate(eval_loader):
        #         eval_batch = prepare_batch_data(eval_batch, local_batch_size)
        #         metrics = p_eval_step(state, eval_batch)
        #         eval_metrics.update(metrics)

        #         if (n_eval_batch + 1) % training_config.log_per_step == 0:
        #             log_for_0("eval: {}/{}".format(n_eval_batch + 1, steps_per_eval))

        #     # compute and log metrics
        #     summary = eval_metrics.compute()
        #     summary = {f"eval_{key}": val for key, val in summary.items()}
        #     summary.update({"ep": ep, "step": step})
        #     logger.log(step + 1, summary)
            
        # # legacy eval
        # if (epoch + 1) % training_config.eval_per_epoch == 0:
        #     log_for_0("Eval epoch {}...".format(epoch))
        #     # sync batch statistics across replicas
        #     state = sync_batch_stats(state)
        #     eval_metrics = MyMetrics(reduction="avg")

        #     for n_eval_batch, eval_batch in enumerate(eval_loader):
        #         eval_batch = prepare_batch_data(eval_batch, local_batch_size)
        #         metrics = p_eval_step(state, eval_batch)
        #         eval_metrics.update(metrics)

        #         if (n_eval_batch + 1) % training_config.log_per_step == 0:
        #             log_for_0("eval: {}/{}".format(n_eval_batch + 1, steps_per_eval))

        #     # compute and log metrics
        #     summary = eval_metrics.compute()
        #     summary = {f"eval_{key}": val for key, val in summary.items()}
        #     summary.update({"ep": ep, "step": step})
        #     logger.log(step + 1, summary)

        # save checkpoint
        if (
            (epoch + 1) % training_config.checkpoint_per_epoch == 0
            or epoch == training_config.num_epochs
        ):
            state = sync_batch_stats(state)
            save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()

    return state

def just_evaluate(config: ml_collections.ConfigDict, workdir: str) -> TrainState:
    ######################################################################
    #                       Initialize training                          #
    ######################################################################
    training_config = config.training
    if jax.process_index() == 0 and training_config.wandb:
        wandb.init(project="sqa_simclr_eval", dir=workdir, tags=["linear"])
        wandb.config.update(config.to_dict())
        ka = re.search(r"kmh-tpuvm-v[234]-(\d+)(-preemptible)?-(\d+)", workdir).group()
        wandb.config.update({"ka": ka})

    rank = jax.process_index()

    logger = GoodLogger(use_wandb=training_config.wandb)

    rng = random.key(training_config.seed)
    
    global_batch_size = training_config.batch_size
    log_for_0("config.batch_size: {}".format(global_batch_size))

    if global_batch_size % jax.process_count() > 0:
        raise ValueError("Batch size must be divisible by the number of processes")
    local_batch_size = global_batch_size // jax.process_count()
    log_for_0("local_batch_size: {}".format(local_batch_size))
    log_for_0("jax.local_device_count: {}".format(jax.local_device_count()))

    if local_batch_size % jax.local_device_count() > 0:
        raise ValueError(
            "Local batch size must be divisible by the number of local devices"
        )
    ######################################################################
    #                           Create Dataloaders                       #
    ######################################################################
    linear_eval_loader, steps_per_linear_eval = input_pipeline.create_split(
        config.dataset,
        local_batch_size,
        split="linear_eval",
    )
    eval_loader, steps_per_eval = input_pipeline.create_split(
        config.dataset,
        local_batch_size,
        split="val",
    )

    log_for_0("steps_per_linear_eval: {}".format(steps_per_linear_eval))
    log_for_0("steps_per_eval: {}".format(steps_per_eval))

    ######################################################################
    #                       Create Train State                           #
    ######################################################################

    lr_scaling = config.training.lr_scaling
    lr = config.training.learning_rate
    if lr_scaling == "linear":
        base_learning_rate = lr * global_batch_size / 256.0
    elif lr_scaling == "sqrt":
        base_learning_rate = lr * jnp.sqrt(global_batch_size)

    net_type = config.model.name
    # get corresponding hidden_dim
    if net_type == "ResNet50": hidden_dim = 2048
    elif net_type == "_ResNet1": hidden_dim = 64
    else: raise NotImplementedError(f"model {net_type} not implemented")
    model = SimCLR(net_type=net_type, hidden_dim=hidden_dim)

    learning_rate_fn = lambda x: base_learning_rate # const lr

    state = create_train_state(
        rng, config, model, config.dataset.image_size, learning_rate_fn
    )
    assert config.load_from is not None
    assert os.path.exists(config.load_from), "checkpoint not found. You should check GS bucket"
    log_for_0("Restoring from: {}".format(config.load_from))
    state = restore_checkpoint(state, config.load_from) # restore checkpoint

    # state = jax_utils.replicate(state)

    p_representation, head_state = init_eval(state, config, hidden_dim, learning_rate_fn, model)

    head_state = jax_utils.replicate(head_state)

    p_linear_eval_step = jax.pmap(
        partial(linear_eval_step, 
                rng_init=rng,
                num_classes=config.dataset.num_classes
        ),
        axis_name="batch",
    )
    p_eval_step = jax.pmap(
        partial(eval_step, num_classes=config.dataset.num_classes),
        axis_name="batch",
    )
    log_for_0("Initial compilation, this might take some minutes...")

    ######################################################################
    #                           Training Loop                            #
    ######################################################################
    timer = Timer()
    for epoch in range(0, training_config.num_epochs):
        if jax.process_count() > 1:
            linear_eval_loader.sampler.set_epoch(epoch)
        log_for_0("epoch {}...".format(epoch))

        # training
        train_metrics = MyMetrics(reduction="last")
        for n_batch, batch in enumerate(linear_eval_loader):
            batch = prepare_linear_eval_batch_data(batch, p_representation)

            head_state, metrics = p_linear_eval_step(head_state, batch)
            train_metrics.update(metrics)

            if epoch == 0 and n_batch == 0:
                log_for_0("Initial compilation completed. Reset timer.")
                timer.reset()

            step = epoch * steps_per_linear_eval + n_batch
            ep = epoch + n_batch / steps_per_linear_eval
            if training_config.get("log_per_step"):
                if (step + 1) % training_config.log_per_step == 0:
                    # compute and log metrics
                    summary = train_metrics.compute_and_reset()
                    summary = {f"train_{k}": v for k, v in summary.items()}
                    summary["steps_per_second"] = (
                        training_config.log_per_step / timer.elapse_with_reset()
                    )
                    summary.update({"ep": ep, "step": step})
                    logger.log(step + 1, summary)

        # evaluation
        if (epoch + 1) % training_config.eval_per_epoch == 0:
            log_for_0("Eval epoch {}...".format(epoch))
            # sync batch statistics across replicas
            # state = sync_batch_stats(state)
            eval_metrics = MyMetrics(reduction="avg")

            for n_eval_batch, eval_batch in enumerate(eval_loader):
                eval_batch = prepare_linear_eval_batch_data(eval_batch, p_representation, local_batch_size)
                metrics = p_eval_step(head_state, eval_batch)
                eval_metrics.update(metrics)

                if (n_eval_batch + 1) % training_config.log_per_step == 0:
                    log_for_0("eval: {}/{}".format(n_eval_batch + 1, steps_per_eval))

            # compute and log metrics
            summary = eval_metrics.compute()
            summary = {f"eval_{key}": val for key, val in summary.items()}
            summary.update({"ep": ep, "step": step})
            logger.log(step + 1, summary)

        # save checkpoint
        if (
            (epoch + 1) % training_config.checkpoint_per_epoch == 0
            or epoch == training_config.num_epochs
        ):
            # state = sync_batch_stats(state)
            save_checkpoint(head_state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()

    return head_state