#!/bin/bash

#====================================================
# SLURM DIRECTIVES
#====================================================
#SBATCH --job-name=HY_MM_FT         # Job name
#SBATCH --output=logs/mm_ft_%j.out   # Standard output log
#SBATCH --error=logs/mm_ft_%j.err    # Standard error log
#SBATCH --time=1-00:00:00            # Maximum wall time
#SBATCH --nodes=2                    # Request 2 nodes
#SBATCH --ntasks-per-node=4          # Request 4 task (process) per node
#SBATCH --gpus-per-node=a100-80:4
#SBATCH --mem=0                      # Allocate memory per node (adjust as needed)
#SBATCH -c 32

#====================================================
# Environment Setup
#====================================================
mkdir -p logs
export NCCL_ASYNC_ERROR_HANDLING=1            # Recommended for NCCL stability
export NCCL_SOCKET_IFNAME=$(ip route get 192.168.51.1 | awk '{print $3; exit}')
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
export MASTER_PORT=29500
export GPUS_PER_NODE=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

#====================================================
# Model Setup
#====================================================

# Training script for HunyuanImage-3.0 multimodal model
# This script supports text-only, text+image, and image generation training

# Model and tokenizer paths - UPDATE THESE TO YOUR PATHS
export MODEL_PATH="/tmp/HunyuanImage-3"
export TOKENIZER_PATH=${MODEL_PATH} # "/tmp/HunyuanImage-3"

export TRAIN_DATA="./example_data/t2i/pretrain.jsonl"
export OUTPUT_DIR="./logs/image_pretrain_hunyuan_multimodal_ft"

# Training configuration
export PER_DEVICE_BATCH_SIZE=4
export GRADIENT_ACCUMULATION_STEPS=1
# NUM_TRAIN_EPOCHS=3
export NUM_TRAIN_STEPS=500
export NUM_STEPS_PER_SAVE=20
export LEARNING_RATE=1e-5
export MIN_LR=5e-6
export MAX_SEQ_LENGTH=8192
export SEQUENCE_TEMPLATE="instruct"

export TEXT_LOSS_WEIGHT=1.0
export IMAGE_GEN_LOSS_WEIGHT=1.0
export WEIGHTING_SCHEME="none"
export LOGIT_MEAN=0.0
export LOGIT_STD=1.0
export MODE_SCALE=1.29

export IMAGE_ROOT_DIR="./example_data/t2i/naruto_images"
export IMAGE_BASE_SIZE=1024
export IMAGE_RATIO="1:1"

export TRAIN_TEXT_ONLY=false
export TRAIN_IMAGE_UNDERSTANDING=false
export TRAIN_IMAGE_GENERATION=true

# Model configuration
export USE_FLASH_ATTN=false
export USE_LORA=true
export LORA_RANK=128
export LORA_ALPHA=256
export LORA_DROPOUT=0.05

# Components to train
export TRAIN_VISION_MODEL=false  # Whether to train ViT encoder
export TRAIN_VAE=false           # Whether to train VAE decoder

# DeepSpeed config (relative to train directory)
export DS_CONFIG="./ds_zero3_no_offload.json"

# Create output directory
mkdir -p $OUTPUT_DIR
export current_time=$(date "+%Y.%m.%d-%H.%M.%S")
export log_file=${OUTPUT_DIR}/"log_${current_time}.txt"

# Download HunyuanImage-3.0 on all nodes
#srun hf download tencent/HunyuanImage-3.0 --local-dir /tmp/HunyuanImage-3

srun -l --jobid $SLURM_JOBID bash -c 'python -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $SLURM_NNODES \
    --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    train_multimodal.py \
        --deepspeed $DS_CONFIG \
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
        --min_lr $MIN_LR \
        --lr_scheduler_type cosine_with_min_lr \
        --warmup_ratio 0.01 \
        --weight_decay 0.1 \
        --logging_steps 1 \
        --save_strategy "steps" \
        --save_total_limit 50 \
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
        --report_to "wandb" \
        --dataloader_num_workers 0'
