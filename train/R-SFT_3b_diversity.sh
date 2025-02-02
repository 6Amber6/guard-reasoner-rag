device="0,1,2,3"
save_path="saves/Llama-3.2-3B/full/guardreasoner_rsft_3b_1"

CUDA_VISIBLE_DEVICES=$device llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset WildGuardTrainR,BeaverTailsTrainR,ToxicChatTrainR \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 16 \
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
    --deepspeed cache/ds_z3_config.json 


save_path="saves/Llama-3.2-3B/full/guardreasoner_rsft_3b_2"

CUDA_VISIBLE_DEVICES=$device llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset WildGuardTrainR,AegisTrainR,ToxicChatTrainR \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 16 \
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
    --deepspeed cache/ds_z3_config.json 



save_path="saves/Llama-3.2-3B/full/guardreasoner_rsft_3b_3"

CUDA_VISIBLE_DEVICES=$device llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path meta-llama/Llama-3.2-3B \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template llama3 \
    --flash_attn auto \
    --dataset_dir data \
    --dataset WildGuardTrainR,AegisTrainR,BeaverTailsTrainR \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 16 \
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
    --deepspeed cache/ds_z3_config.json 

