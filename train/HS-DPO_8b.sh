device="0,1,2,3"
save_path="saves/Llama-3.1-8B/full/guardreasoner_hsdpo_8b"

CUDA_VISIBLE_DEVICES=$device llamafactory-cli train \
    --stage dpo \
    --do_train True \
    --model_name_or_path "saves/Llama-3.1-8B/full/guardreasoner_rsft_8b" \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset WildGuardTrainHS_8B,AegisTrainHS_8B_merge,BeaverTailsTrainHS_8B_merge,ToxicChatTrainHS_8B_merge \
    --cutoff_len 2048 \
    --learning_rate 5e-06 \
    --num_train_epochs 2.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir $save_path \
    --bf16 True \
    --plot_loss True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --pref_beta 0.01 \
    --pref_ftx 2.0 \
    --pref_loss sigmoid \
    --deepspeed cache/ds_z3_config.json 
