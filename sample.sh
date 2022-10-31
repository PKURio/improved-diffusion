MODEL_FLAGS="--image_size 128 --batch_size 50 --num_samples 400 --class_cond True --class_type 2"
DIFFUSION_FLAGS="--diffusion_steps 4000 --noise_schedule linear"
NUM_GPUS=8
MODEL_PATH="logs/conditional_clear/ema_0.9999_300000.pt"

export PYTHONPATH=/ssd/liaoran/improved-diffusion
export OPENAI_LOGDIR=/ssd/liaoran/improved-diffusion/logs/conditional_clear/samples
mpiexec -n $NUM_GPUS python3 scripts/image_sample.py --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
