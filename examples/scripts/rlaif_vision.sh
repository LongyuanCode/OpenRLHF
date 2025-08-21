#!/bin/bash
#RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --working-dir . -- \
python openrlhf/cli/rlaif_train_ray.py \
    --dataset_name "OpenBMB/RLAIF-V-Dataset" \
    --max_data_num 5000 \
    --is_put_images_object_store False \
    --prefered_inferior_pair_num 2 \
    --num_labeler_vllm_engines 2 \
    --num_policy_vllm_engines 2 \
    --enforce_eager_labeler \
    --enforce_eager_policy \
    --gpu_memory_utilization_labeler 0.9 \
    --vllm_enable_sleep_labeler False \
    --enable_prefix_caching_policy False \
    --gpu_memory_utilization_policy 0.9 \
    --vllm_enable_sleep_policy False \
    --enable_prefix_caching_labeler False \
    --use_ds_universal_ckpt \
    --zero_stage 2 \
    --temperature 0.0 \
    --gradient_checkpointing True \
    --gradient_checkpointing_use_reentrant False \
    --deepcompile False \
    --bf16 True \
    --labeler_load_in_8bit False \
    --labeler_load_in_4bit False \
    --labeler_bf16 True \
    --policy_load_in_4bit False \
    --policy_load_in_8bit False \
    --policy_bf16 True \
    --use_flash_attn_policy True \
    --pretrain_labeler "/root/gpufree-data/modelscope_cache/models/llava-hf/llava-1.5-13b-hf" \
    --seed_labeler 42 \
    --full_determinism True \
    --pretrain_policy "/root/gpufree-data/modelscope_cache/models/llava-hf/llava-1.5-7b-hf" \
    --seed_policy 42 \
    --n_candidates 5 \
    --max_new_tokens 256 \
    --policy_generate_temperature 0.7 \
    --do_sample True \
    --max_len 1024 \
    --prompt_max_len 600 \
    --generate_max_len 256 \
    --labeler_batch_size 100 \
    --num_epochs 1 \
    --num_workers 4 \
    --num_gpus_per_node 4 \
    --num_gpus_per_policy_actor 1 \
    --num_gpus_per_ref_actor 1 \
    --num_nodes 1 \
    --vllm_tensor_parallel_size_labeler 1 \
    --vllm_tensor_parallel_size_policy 1 \
    --enable_prefix_caching_labeler False \
    --freeze_vision_tower True \
    --vision_tower_name "vision_tower" \
    \
    --actor_learning_rate 1e-5 \
    --lr_scheduler "cosine_with_min_lr" \
    --lr_warmup_ratio 0.1 \
    --adam_betas 0.9 0.999 \
    --l2 0.0 \
    \
    --policy_train_bf16 True \
    --policy_train_load_in_4bit False \
    --policy_train_load_in_8bit False \
    \
    --disable_ds_ckpt True \
    --ckpt_path "./ckpt" \
    --load_checkpoint True \
    --max_ckpt_num 3 \
    --max_ckpt_mem 20.0 \
    --save_path "./trained_policy_model" \
    \
    --deepspeed_enable_sleep False \
    --ds_tensor_parallel_size_policy 1 \
    --ds_tensor_parallel_size_ref 1 \
    --ring_attn_size 1 \
    --train_batch_size 2 \
    --micro_train_batch_size 1 \
    \
    --vllm_sync_backend "nccl" \
    --vllm_sync_with_ray False \
    --colocate_all_models False \
    \
    --wandb_enable False
