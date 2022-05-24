MODEL_FLAGS=""
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 100"

export PYTHONPATH=/home/liaoran/improved-diffusion
python3 scripts/new_train.py $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS

