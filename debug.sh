source config.sh
rm -rf tmp # Comment this line if you want to reload (usually not the case)

CONDA_PATH=$(which conda)
echo "find conda path:" $CONDA_PATH
CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
LOGDIR=$(pwd)/tmp
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}

source $CONDA_INIT_SH_PATH
# remember to use your own conda environment
conda activate $OWN_CONDA_ENV_NAME

echo "start running main"

# JAX_PLATFORMS=cpu python3 main.py \
#     --workdir=${LOGDIR} \
#     --mode=local_debug \
#     --config=configs/load_config.py:local_debug \
#     --config.dataset.root=${EU_IMAGENET_FAKE} \
# 2>&1 | grep --invert-match Could

python3 main.py \
    --workdir=${LOGDIR} \
    --mode=local_debug \
    --config=configs/load_config.py:local_debug \
    --config.dataset.root=${EU_IMAGENET_FAKE} \
    --debug
2>&1 | grep --invert-match Could