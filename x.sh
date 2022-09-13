MODEL_FLAGS="--image_size 64"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 10"
NUM_GPUS=8
DATA_DIR="/home/zhengyue/pcb_patch_dataset_checked/划伤/严重"

export PYTHONPATH=/home/liaoran/improved-diffusion
mpiexec -n $NUM_GPUS python3 scripts/new_train.py --data_dir "/home/zhengyue/pcb_patch_dataset_checked/划伤/严重" $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

