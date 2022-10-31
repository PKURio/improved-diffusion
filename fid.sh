BATCH=50
DATA_DIR="/ssd/liaoran/pcb_patch_dataset_checked/qy/clear"
GENERATE_DIR="/ssd/liaoran/improved-diffusion/logs/conditional_clear2/samples/samples_400x128x128x3_2.npz"

export PYTHONPATH=/ssd/liaoran/improved-diffusion
python3 fid/fid_score.py $DATA_DIR $GENERATE_DIR --batch $BATCH --device cuda:0

