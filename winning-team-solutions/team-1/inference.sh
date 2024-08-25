#!/bin/bash 

SEED=42
BACKBONE=convnext_tiny
CKPT_PATH="/u/yashjain/kaggle_4/winning-team-solutions/team-1/model_weights/convnext_tiny_1536_customloss_e20.pth"
IN_CHANNELS=3
NUM_CLASSES=3
IMAGE_SIZE=3072
BATCH_SIZE=2
THRESHOLD=0.4
AXIS="z|y|x"
FLIP=5
ROT=3

for group in kidney_6 kidney_5; do
    python inference.py \
    --seed $SEED \
    --group $group \
    --backbone $BACKBONE \
    --ckpt_path $CKPT_PATH \
    --in_channels $IN_CHANNELS \
    --num_classes $NUM_CLASSES \
    --image_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --axis $AXIS \
    --flip $FLIP \
    --rot $ROT \
    --overlap \
    --threshold $THRESHOLD
done
