# sqa added: SimCLR

*representation dim*: 
- For ResNet50 with ImageNet, 2048
- For _ResNet1 with ImageNet, 64

Pretrained checkpoints:
- ResNet 50, 20250225: `/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20250224_200900_f7hn8q_kmh-tpuvm-v3-32-1__b_lr_ep_torchvision_r50_eval/checkpoint_62500`

- ResNet 50, 20250227, fixed MLP: `/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20250225_232107_8qpogt_kmh-tpuvm-v3-32-1__b_lr_ep_torchvision_r50_eval/checkpoint_62500`

- ResNet 50, 20250228, fixed BN and bias: `/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20250227_225128_fl5x11_kmh-tpuvm-v3-32-preemptible-1__b_lr_ep_torchvision_r50_eval/checkpoint_62500`

- ResNet 50, 20250301, weight decay mask: `/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20250301_004259_wuz8n9_kmh-tpuvm-v3-32-preemptible-1__b_lr_ep_torchvision_r50_eval/checkpoint_62500`

based on:

# JAX+linen Development Codebase for Image Classification

In this repository, we provide a codebase for training ResNet50+ImageNet on JAX+linen.

## Setting up the environment

Don't forget to add your `config.sh`: 

```shell
export EU_IMAGENET_FAKE="/kmh-nfs-ssd-eu-mount/code/hanhong/data/imagenet_fake/kmh-nfs-ssd-eu-mount/" # path to the fake imagenet data at EU
export US_IMAGENET_FAKE="/kmh-nfs-ssd-eu-mount/code/hanhong/data/imagenet_fake/kmh-nfs-us-mount/" # path to the fake imagenet data at US
export OWN_CONDA_ENV_NAME=NNX # A useable env at your V3-8
export WANDB_API_KEY= # Your wandb API key
export TASKNAME=resnet_linen # Task name
```

## Expected Performance

| Card | Run Time | Accuracy | Step per sec | url |
| --- | --- | --- | --- | --- |
| V3-32 | 3h 28m 56s | 76.15 | ~11 | [eager-sea-31](https://wandb.ai/zhh24-massachusetts-institute-of-technology/resnet_linen/runs/7rnmj3jx) |
| V2-32 | 3h 38m 50s | 76.21 | ~10.5 | [fresh-deluge-34](https://wandb.ai/zhh24-massachusetts-institute-of-technology/resnet_linen/runs/nemvjmm5) |

## Run the code

1. Setup your `config.sh`.

2. Select your TPU in `ka.sh`.

3. Run

```shell
source staging.sh
```

# The original README

# Jax training with PyTorch dataloaders

Written by Congyue Deng, Kaiming He.

### Notes
PyTorch DataLoader

- This branch is built on top of the main branch: https://github.com/KaimingHe/resnet_jax/tree/main, which is based on the TFDS dataloader.

- For debugging, use the flag `--debug=True` to call `with jax.disable_jit():` which disables jax compilation. Be careful, this may increase memory consumption

- Increase `num_workers` and `prefetch_factor`. A thorough discussion: https://github.com/pytorch/xla/issues/2690

## Step-by-Step Instruction

### Introduction

Before you start, please read [He Vision Group's TPU documentation](https://docs.google.com/document/d/1khhWZW7VJ8HmrzaQIYcHRKlLygaeceHHLdWHoakwbdU/edit).

JAX and TPU are useful and great. But GCP (Google Cloud Platform) takes some time to learn. This tutorial will walk you through some basic concepts about JAX, TPU, and GCP.

### SSH into your TPU VM

SSH into your TPU VM that has 8 TPUs (say, v4-8). Every 8 TPUs are in one node. Unlike PyTorch, in JAX, we only need one Python process per node. This **single-node** TPU VM works as your dev machine.

If you have followed [He Vision Group's TPU documentation](https://docs.google.com/document/d/1khhWZW7VJ8HmrzaQIYcHRKlLygaeceHHLdWHoakwbdU/edit), you should be able to SSH into a TPU VM by running the following command in your **laptop**:
```shell
DEV_VM=kmh-tpuvm-v4-8-1
ssh $DEV_VM
```
Here `kmh-tpuvm-v4-8-1` is the TPU VM's name. In the following, we assume you are already in your dev TPU VM.

Check this "[manual SLRUM](https://docs.google.com/spreadsheets/d/1vDpP7eTkYRwWYs2fo-9dJwxHKvN6SDp5iC1cM1H3FDU/edit?usp=sharing)" spreadsheet for available TPU VMs.

### Mount NFS Filestore

All our TPU VMs will access to a shared file system for data and code. The TPU VMs are not permanent; the shared file system is. We use NFS Filestore from GCP. In your TPU VM, run the following to install the tool for mounting:
```shell
sudo apt-get -y update
sudo apt-get -y install nfs-common
```
Then you can mount a disk by:
```shell
sudo mkdir -p /kmh-nfs-us-mount
sudo mount -o vers=3 10.26.72.146:/kmh_nfs_us /kmh-nfs-us-mount
sudo chmod go+rw /kmh-nfs-us-mount
ls /kmh-nfs-us-mount
```
Here `/kmh-nfs-us-mount` is like a local dir that can be accessed from your TPU VM.

### Mount NFS Filestore (*** with Pytorch Loader ***)

To use Pytorch's dataloader, we need high-throughput SSD disk located in the same region as your TPU VMs. Mount the following SSD disk in `europe-west4-a` when you are using **TPU v3** machines for your remote jobs:
```shell
sudo mkdir -p /kmh-nfs-ssd-eu-mount
sudo mount -o vers=3 10.150.179.250:/kmh_nfs_ssd_eu /kmh-nfs-ssd-eu-mount
sudo chmod go+rw /kmh-nfs-ssd-eu-mount
ls /kmh-nfs-ssd-eu-mount
```

The ImageNet dataset, in their per-image raw formats for Pytorch dataloader, is in `/kmh-nfs-ssd-eu-mount/data/imagenet`.

### Manage your code

We recommend you to put your code in the NFS mount, not in the local TPU VM. Then your code can be run in different machines. Create a dir in the mount and clone this repo:
```
sudo chmod go+rw /kmh-nfs-us-mount/code
mkdir /kmh-nfs-us-mount/code/$USER/
cd /kmh-nfs-us-mount/code/$USER/
git clone https://github.com/KaimingHe/resnet_jax.git
cd resnet_jax
```


### Install packages

Run `pip install -r requirements.txt` to install the minimal requirements.

- torch, torchvision
- [2024.4] ml_collections not working with Python>=3.12 (imp deprecated): https://github.com/google/ml_collections/pull/28. To fix this problem, install ml_collections from:
  ```
  pip install git+https://github.com/danielkelshaw/ml_collections.git
  ```

### Run single-node training (*** with Pytorch Loader ***)

In your TPU VM, run the following for sanity check of JAX/TPU:
```shell
python3 -c 'import jax; print(jax.device_count())'
```
It should output `4` for TPU v4-8 (or `8` for TPU v3-8). If you are waiting forever, it implies you are not in a single-node TPU VM; if you see some error with `/dev/accel0`, it implies the TPUs are run by someone else.

Now, run a single-node training:
```shell
PWD=$(pwd)
python3 main.py \
    --workdir=${PWD}/tmp --config=configs/tpu.py \
    --config.dataset.cache=True \
    --config.dataset.root=./imagenet_fake \
    --config.batch_size=1024 \
    --config.dataset.prefetch_factor=2 \
    --config.dataset.num_workers=32 \
    --config.log_per_step=10 \
    --config.model='_ResNet1'
```

This command can also be found in `run_script.sh`, which is what I use to run local dev jobs.

**Note:**
- `./imagenet_fake` contains just soft links to the `/kmh-nfs-us-mount/data/imagenet/val` dir: **both train and val in are validation sets**, only for fast debugging.
- `_ResNet1` is a tiny ResNet for fast debugging.

The first few iterations of the log look like this:
<img width="1155" src="https://github.com/KaimingHe/deep-residual-networks/assets/11435359/a181d0f2-591f-47f9-9c87-f884feb0ee92">


You can see that the speed is not ideal, even though we train a tiny `_ResNet1`. **Data loading time with Pytorch loader is a major bottleneck for small models** (including even ResNet-50, 101, 152; see below).


### Run multi-node training

#### Concept

Any TPU VM with more than 8 TPUs is conceptually a multi-node machine. For example, `v3-32` is conceptually 4 nodes, and we will do the same thing for each node. To have some sense of it, run the following:
```
VM_NAME=kmh-tpuvm-v3-32-1
ZONE=europe-west4-a

gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "echo HelloWorld"
```
And you will see:
```
Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 4 workers.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
HelloWorld
HelloWorld
HelloWorld
HelloWorld
```

**Note:**
- When you run this demo, make sure `kmh-tpuvm-v3-32-1` is available.


#### Install packages in remote nodes

We need to install all packages and mount NFS in our remote TPU VM:

Open `run_init_remote.sh`, change `VM_NAME` and `ZONE` into your remote TPU VM, say: `VM_NAME=kmh-tpuvm-v3-32-1` and ` ZONE=europe-west4-a`.

Then run `source run_init_remote.sh`. This will install packages and mount NFS in the remote TPU VM.


#### Manage your jobs

The "remote" TPU VM is like your dev TPU VM. Conceptually, we need to run the same code for all nodes in the multi-node TPU VM, one Python process per node. So first of all, we need nodes to access to the same copy of code, and the same destination of artifacts (logs and checkpoints).

In your **dev** TPU VM (say, `v4-8`), run:
```
mkdir /kmh-nfs-us-mount/logs/$USER/
mkdir /kmh-nfs-us-mount/staging/$USER/
sudo chmod 777 /kmh-nfs-us-mount/logs/$USER
sudo chmod 777 /kmh-nfs-us-mount/staging/$USER
```
Here, `logs` is the dir to the remote job's artifacts, and `staging` is the dir for staged (cached) **copies** of codes that wil be run in remote TPU VM.

**Note:**
- You may notice that here the artifacts are in `/kmh-nfs-us-mount` (zone=`us`). In case you have big artifacts (e.g., very large checkpoints), you may want to use the mount in the same zone as your remote TPU VM (`/kmh-nfs-ssd-eu-mount` for TPU v3 in zone=`eu`):
```
mkdir /kmh-nfs-ssd-eu-mount/logs/$USER/
sudo chmod 777 /kmh-nfs-ssd-eu-mount/logs/$USER
```


#### Run a remote job

You may open `run_remote.sh` to see how a remote job is run. Conceptually, the essential part is (you don't need to run this line):
```
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "python3 main.py"
```
Here, `--worker=all` means the same command `python3 main.py` will be run in all nodes.

The file `run_remote.sh` alone does not take effect; instead, we use `run_staging.sh` to kick off a remote job. You may open `run_staging.sh` and see the process. Basicall, it will copy the current repo dir (in you dev TPU VM) to a hashed dir in `/kmh-nfs-us-mount/staging/$USER/` and `cd` into it, then it will run the `run_remote.sh` file in the staging dir.

In sum, you only need to run `run_staging.sh` in your **dev** TPU VM by:
```
source run_staging.sh
```
Then you can kick off your remote job.

The following is the beginning of the output you may see:
```
Staging files...
Done staging.
Current dir: /kmh-nfs-us-mount/staging/kaiminghe/240419000859-u1fe8t-2df5ac0-code
kmh-tpuvm-v3-32-2 europe-west4-a
Log dir: /kmh-nfs-ssd-eu-mount/logs/kaiminghe/resnet/20240419_000859_d4awst_kmh-tpuvm-v3-32-2_tpu_b1024_lr0.1_ep100_torchvision_ep1000x
Using ssh batch size of 1. Attempting to SSH into 1 nodes with a total of 4 workers.
SSH: Attempting to connect to worker 0...
SSH: Attempting to connect to worker 1...
SSH: Attempting to connect to worker 2...
SSH: Attempting to connect to worker 3...
Current dir: /kmh-nfs-us-mount/staging/kaiminghe/240419000859-u1fe8t-2df5ac0-code
Current dir: /kmh-nfs-us-mount/staging/kaiminghe/240419000859-u1fe8t-2df5ac0-code
Current dir: /kmh-nfs-us-mount/staging/kaiminghe/240419000859-u1fe8t-2df5ac0-code
Current dir: /kmh-nfs-us-mount/staging/kaiminghe/240419000859-u1fe8t-2df5ac0-code
```

#### Cancel a remote job (NOTE: NOT DELETING TPU VM)

Oftentimes you may not want your remote job to finish training (e.g., you found a bug in your code). You may Ctrl+C with your remote job, but it only shuts down the ssh client running it, not the the job itself. If you will run anothe job in the same remote TPU VM, you may see some error related to `/dev/accel0`, which implies TPUs have been running by some other jobs.

When this happens, you may use `run_kill_remote.sh` to kill the jobs in remote TPU VM. Run `source run_kill_remote.sh` in your **dev** TPU VM.

**Caution 1**: Before you kill the job, make sure the remote TPU VM is the one you want to handle. Running this `run_kill_remote.sh` on another TPU VM can kill other people's jobs.

**Caution 2**: This command in `run_kill_remote.sh` is only about killing a job. It won't delete the TPU VM. And we suggest NOT to delete TPU VM unless necessary.


### Tensorboard Monitoring

To access the Tensorboard profile generated by the job, run the following in your **laptop** when SSH into your dev TPU VM:
```shell
ssh $DEV_VM -L 6060:localhost:6060
```
You may change `6060` into whatever number you like.

Assume Tensorboard has been installed in the dev TPU VM. If the alias of `tensorboard` is not defined, you may run `alias tensorboard='python3 -m tensorboard.main'` to define it (you may want to put this into your `.zshrc` or `.bashrc`).

In your TPU VM, run the following:
```
tensorboard --port=6060 --logdir_spec=\
v3-32-2_tpu_b1024_lr0.1_ep100_torchvision:/kmh-nfs-ssd-eu-mount/logs/kaiminghe/resnet/20240419_020919_nevsyj_kmh-tpuvm-v3-32-2_tpu_b1024_lr0.1_ep100_torchvision
```
Then open `http://localhost:6060/#scalars` in your laptop. You can see the tensorboard profile:

<img width="587" src="https://github.com/KaimingHe/deep-residual-networks/assets/11435359/39829a1f-6258-4062-aed3-71fdc88891f5">


**Note:**
- Here, the metrics starting with `ep_` will have `epochs` (x1000) as the x-axis. For example, x-axis with 100k just means 100 epochs. This is useful for calibrating different runs with different batch sizes. 


Here is another run with 128 TPUs (`v3-128`, 16 nodes):
```
tensorboard --port=6060 --logdir_spec=\
v3-128-1_tpu_b8192_lr0.1_ep100_torchvision:/kmh-nfs-ssd-eu-mount/logs/kaiminghe/resnet/20240419_023004_itp67c_kmh-tpuvm-v3-128-1_tpu_b8192_lr0.1_ep100_torchvision
```

It is about 3x faster than `v3-32`. Note that data loading is still the bottleneck.
<img width="610" alt="Screenshot 2024-04-19 at 10 17 22 AM" src="https://github.com/KaimingHe/deep-residual-networks/assets/11435359/3592e263-d509-4d62-a087-28c6c3c8621f">


### *** When to use Pytorch Loader? ***

- As you may see, when using the Pytorch Loader, training ResNet-50 is 2-3x slower than using TFDS. This is because data loader time is the bottleneck.

- Because of this, training ResNet-50, -101, -152, -200 basically has the same time. See below, with 128 TPUs (`v3-128`) and a batch size of 4096. The training time hasn't overlapped the loading time until ResNet-200.

<img width="621" src="https://github.com/KaimingHe/deep-residual-networks/assets/11435359/12e96992-53d3-4fa6-b8fe-41f9043b9f12">


- Based on this observation, Pytorch dataloader is most favored when training **large** models (like ViT), and when using fancy data augmentation (like [timm](https://pypi.org/project/timm/)) which is readily available in Pytorch.

### *** Numerical Reproducibility ***

In JAX, the random seed is treated as part of the input to the function; as such, the JAX computation is fully deterministic. In TPUs, the computation is fully deterministic (while in GPU it is not). Using Pytorch Loader, the multi-process dataloader (in both single-/multi-node settings) has process/worker-specific random seed control, and can be made fully deterministic.

When using the same random seed, the same versions of packages, and the same hyper-parameters, running the same job would give **100% numerically exact same result**, up to every single digit and every iteration.

Below are the training loss curves of two jobs. They are exactly the same. Even the running time is very similar.

<img width="1159" src="https://github.com/KaimingHe/deep-residual-networks/assets/11435359/b9a58d7d-69cd-45b8-b419-79eda6a0b062">
