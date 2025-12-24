#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Training script for HunyuanImage-3.0 multimodal model
# This script supports text-only, text+image, and image generation training

# Model and tokenizer paths
MODEL_PATH="/tmp/HunyuanImage-3"
TOKENIZER_PATH=${MODEL_PATH} # "/tmp/HunyuanImage-3"

# Data path
TRAIN_DATA="./example_data/t2i/pretrain.jsonl"

# Output directory
OUTPUT_DIR="./logs/image_pretrain_hunyuan_multimodal_ft"

# Training configuration
PER_DEVICE_BATCH_SIZE=2
GRADIENT_ACCUMULATION_STEPS=1
# NUM_TRAIN_EPOCHS=3
NUM_TRAIN_STEPS=200
NUM_STEPS_PER_SAVE=50
LEARNING_RATE=1e-5
MIN_LR=5e-6
MAX_SEQ_LENGTH=8192
SEQUENCE_TEMPLATE="instruct"

# Train image configuration
TEXT_LOSS_WEIGHT=1.0
IMAGE_GEN_LOSS_WEIGHT=1.0
WEIGHTING_SCHEME="none"
LOGIT_MEAN=0.0
LOGIT_STD=1.0
MODE_SCALE=1.29

IMAGE_ROOT_DIR="./example_data/t2i/naruto_images"
IMAGE_BASE_SIZE=256
IMAGE_RATIO="1:1"

# Training modes
TRAIN_TEXT_ONLY=false            # Language modeling on text
TRAIN_IMAGE_UNDERSTANDING=false  # Vision-language understanding
TRAIN_IMAGE_GENERATION=true      # Image generation

# Model configuration
USE_FLASH_ATTN=false
USE_LORA=true
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
USE_QLORA=true
LOAD_IN_4BIT=true
LOAD_IN_8BIT=false
MOE_DROP_TOKENS=false

# Components to train
TRAIN_VISION_MODEL=false  # Whether to train ViT encoder
TRAIN_VAE=false           # Whether to train VAE decoder

# DeepSpeed config (relative to train directory)
# DS_CONFIG="./configs/ds_zero3_no_offload.json"

# Create output directory
mkdir -p $OUTPUT_DIR
current_time=$(date "+%Y.%m.%d-%H.%M.%S")
log_file=${OUTPUT_DIR}/"log_${current_time}.txt"


# Run training
deepspeed --num_gpus=1 train_multimodal.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --train_data_file $TRAIN_DATA \
    --output_dir $OUTPUT_DIR \
    --overwrite_output_dir \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --max_steps $NUM_TRAIN_STEPS \
    --save_steps $NUM_STEPS_PER_SAVE \
    --optim "paged_adamw_8bit" \
    --min_lr $MIN_LR \
    --lr_scheduler_type cosine_with_min_lr \
    --warmup_ratio 0.01 \
    --weight_decay 0.1 \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_total_limit 3 \
    --save_safetensors False \
    --bf16 true \
    --tf32 true \
    --max_seq_length $MAX_SEQ_LENGTH \
    --model_max_length $MAX_SEQ_LENGTH \
    --use_flash_attn $USE_FLASH_ATTN \
    --use_lora $USE_LORA \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --use_qlora $USE_QLORA \
    --load_in_4bit $LOAD_IN_4BIT \
    --load_in_8bit $LOAD_IN_8BIT \
    --moe_drop_tokens $MOE_DROP_TOKENS \
    --train_vision_model $TRAIN_VISION_MODEL \
    --train_vae $TRAIN_VAE \
    --train_text_only $TRAIN_TEXT_ONLY \
    --train_image_understanding $TRAIN_IMAGE_UNDERSTANDING \
    --train_image_generation $TRAIN_IMAGE_GENERATION \
    --gradient_checkpointing true \
    --ddp_find_unused_parameters false \
    --text_loss_weight $TEXT_LOSS_WEIGHT \
    --image_gen_loss_weight $IMAGE_GEN_LOSS_WEIGHT \
    --weighting_scheme $WEIGHTING_SCHEME \
    --logit_mean $LOGIT_MEAN \
    --logit_std $LOGIT_STD \
    --mode_scale $MODE_SCALE \
    --image_root_dir $IMAGE_ROOT_DIR \
    --image_base_size $IMAGE_BASE_SIZE \
    --image_ratio $IMAGE_RATIO \
    --sequence_template $SEQUENCE_TEMPLATE \
    --report_to "tensorboard" \
    --dataloader_num_workers 0 | tee ${log_file}
