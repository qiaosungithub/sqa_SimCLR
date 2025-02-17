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
import jax, ml_collections, optax, wandb, re
from jax import lax, random
import jax.numpy as jnp

import input_pipeline
from input_pipeline import prepare_batch_data
# import models.models_resnet as models_resnet
from models.models_simclr import SimCLR

from utils.info_util import print_params
from utils.logging_utils import log_for_0, GoodLogger
from utils.metric_utils import Timer, MyMetrics


def create_model(*, model_cls, half_precision, num_classes, **kwargs):
    """
    Create a model using the given model class.
    """
    platform = jax.local_devices()[0].platform
    if half_precision:
        if platform == "tpu":
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32
    return model_cls(num_classes=num_classes, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
    """
    Initialize the model, and return the model parameters.
    """
    input_shape = (1, image_size, image_size, 3)

    @jax.jit
    def init(*args):
        return model.init(*args)

    log_for_0("Initializing params...")
    variables = init({"params": key}, jnp.ones(input_shape, model.dtype))
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


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
    """
    Create initial training state.
    """
    platform = jax.local_devices()[0].platform
    if config.model.half_precision and platform == "gpu":
        raise NotImplementedError("We consider TPU only.")

    params, batch_stats = initialized(rng, image_size, model)

    log_for_0(params, logging_fn=print_params)

    if config.training.optimizer == "sgd":
        tx = optax.sgd(
            learning_rate=learning_rate_fn,
            momentum=config.training.momentum,
            nesterov=True,
        )
    elif config.training.optimizer == "LARS":
        tx = optax.lars(
            learning_rate=learning_rate_fn,
            weight_decay=config.training.weight_decay,
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
    assert features.ndim == 3
    features = features.reshape(-1, features.shape[-1])
    B = features.shape[0] // 2
    # generate mask
    labels = jnp.concatenate([jnp.arange(B) for _ in range(2)], axis=0)
    labels = (labels[:, None] == labels[None, :]).astype(jnp.float32)
    mask = jnp.eye(2 * B, dtype=jnp.bool)
    labels = labels[~mask].reshape(2 * B, 2 * B - 1)
    # compute similarity matrix
    features = features / jnp.linalg.norm(features, axis=1, keepdims=True)
    similarity_matrix = jnp.dot(features, features.T)
    assert similarity_matrix.shape == (2 * B, 2 * B)
    similarity_matrix = similarity_matrix[~mask].reshape(2 * B, 2 * B - 1)
    # compute logits
    positives = similarity_matrix[labels.astype(jnp.bool)].reshape(2 * B, 1)
    negatives = similarity_matrix[~labels.astype(jnp.bool)].reshape(2 * B, 2 * B-2)

    # print(f"B: {B}, positives shape: {positives.shape}, negatives shape: {negatives.shape}")
    # exit("路明")

    logits = jnp.concatenate([positives, negatives], axis=1)
    logits /= temperature
    # compute loss
    prob = jax.nn.softmax(logits, axis=1)
    prob = prob[:, 0]
    loss = jnp.mean(-jnp.log(prob))
    # compute acc
    pred = jnp.argmax(logits, axis=1)
    acc = jnp.mean(pred == 0)
    return loss, acc

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
        features = lax.all_gather(features, axis_name="batch")
        # assert False, f"features shape: {features.shape}, images shape: {images.shape}" # feature: (8, 2b_2, c), images: (2b_2, 224, 224, 3)
        loss, acc = NTXent(features, temperature=config.training.temperature)
        # loss = cross_entropy_loss(logits, batch["label"], num_classes)
        # weight_penalty_params = jax.tree_util.tree_leaves(params)
        # weight_l2 = sum(jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1)
        # weight_penalty = weight_decay * 0.5 * weight_l2
        # loss = loss + weight_penalty
        return loss, (new_model_state, acc)

    # compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    grads = lax.pmean(grads, axis_name="batch")
    new_model_state, acc = aux[1]

    # apply gradients
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )

    # compute metrics
    # metrics = compute_metrics(logits, batch["label"], num_classes)
    metrics = {"loss": aux[0], "contrastive_acc": acc}
    metrics["lr"] = learning_rate_fn(state.step)
    return new_state, metrics


def eval_step(state, batch, num_classes):
    """
    Perform a single evaluation step. This function will be pmap, so we can't print inside it.
    """
    variables = {"params": state.params, "batch_stats": state.batch_stats}
    logits = state.apply_fn(variables, batch["image"], train=False, mutable=False)
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
    # features = jnp.ones((8, 2, 64))
    # NTXent(features, temperature=0.1)
    ######################################################################
    #                       Initialize training                          #
    ######################################################################
    training_config = config.training
    if jax.process_index() == 0 and training_config.wandb:
        wandb.init(project="sqa_simclr", dir=workdir)
        wandb.config.update(config.to_dict())
        ka = re.search(r"kmh-tpuvm-v[23]-32(-preemptible)?-(\d+)", workdir).group()
        wandb.config.update({"ka": ka})

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
    eval_loader, steps_per_eval = input_pipeline.create_split(
        config.dataset,
        local_batch_size,
        split="val",
    )
    log_for_0("steps_per_epoch: {}".format(steps_per_epoch))
    log_for_0("steps_per_eval: {}".format(steps_per_eval))

    if training_config.steps_per_eval != -1:
        steps_per_eval = training_config.steps_per_eval

    ######################################################################
    #                       Create Train State                           #
    ######################################################################

    base_learning_rate = training_config.learning_rate * global_batch_size / 256.0

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
    state = restore_checkpoint(state, workdir)
    step_offset = int(state.step)
    epoch_offset = step_offset // steps_per_epoch
    assert (
        epoch_offset * steps_per_epoch == step_offset
    ), "Your checkpoint step {} is not aligned with steps_per_epoch {}".format(
        step_offset, steps_per_epoch
    )
    state = jax_utils.replicate(state)

    ######################################################################
    #                     Prepare for Training Loop                      #
    ######################################################################

    p_train_step = jax.pmap(
        partial(
            train_step,
            rng_init=rng,
            learning_rate_fn=learning_rate_fn,
            # num_classes=config.dataset.num_classes,
            weight_decay=training_config.weight_decay,
            config=config,
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
    for epoch in range(epoch_offset, training_config.num_epochs):
        if jax.process_count() > 1:
            train_loader.sampler.set_epoch(epoch)
        log_for_0("epoch {}...".format(epoch))

        # training
        train_metrics = MyMetrics(reduction="last")
        for n_batch, batch in enumerate(train_loader):
            batch = prepare_batch_data(batch)

            # # sanity check
            # def tang(i): return i
            # p_tang = jax.pmap(tang, axis_name="batch")
            # images = batch["image"]
            # print(f"images.shape: {images.shape}")
            # images = p_tang(images)
            # print(f"images.shape: {images.shape}")
            # images = images.reshape(-1, 224, 224, 3)
            # B = images.shape[0] // 2
            # import matplotlib.pyplot as plt
            # import os
            # path = f"/kmh-nfs-ssd-eu-mount/code/qiao/work/dataset_images_sanity/{n_batch}"
            # if os.path.exists(path) == False: os.makedirs(path)
            # for i in range(2*B):
            #     img = images[i]
            #     plt.imsave(path+f"/{i}.png", img)
            # exit("邓东灵")

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

            state, metrics = p_train_step(state, batch)
            train_metrics.update(metrics)

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
        # evaluation
        if (epoch + 1) % training_config.eval_per_epoch == 0:
            log_for_0("Eval epoch {}...".format(epoch))
            # sync batch statistics across replicas
            state = sync_batch_stats(state)
            eval_metrics = MyMetrics(reduction="avg")

            for n_eval_batch, eval_batch in enumerate(eval_loader):
                eval_batch = prepare_batch_data(eval_batch, local_batch_size)
                metrics = p_eval_step(state, eval_batch)
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
            state = sync_batch_stats(state)
            save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.key(0), ()).block_until_ready()

    return state
