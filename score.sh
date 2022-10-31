MODEL_FLAGS="--image_size 128 --class_cond True"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 20"
NUM_GPUS=8
# shellcheck disable=SC2140
DATA_DIR="/ssd/liaoran/pcb_patch_dataset_checked/hs/clear|"\
"/ssd/liaoran/pcb_patch_dataset_checked/ptct/clear|"\
"/ssd/liaoran/pcb_patch_dataset_checked/qy/clear|"\

export PYTHONPATH=/ssd/liaoran/improved-diffusion
export OPENAI_LOGDIR=/ssd/liaoran/improved-diffusion/logs/conditional_clear
mpiexec -n $NUM_GPUS python3 scripts/new_train.py --data_dir $DATA_DIR $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

