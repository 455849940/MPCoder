export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=3,5 torchrun --master_port=65532 --nproc_per_node 2 main.py \
    --do_train True \
    --do_eval True \
    --debug_mode False\
    --learning_rate 1e-4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 10 \
    --choose_model_name perfer_AugT\
    --output_dir augT_model_linear \
    --train_data_path ./data/Java_part_programming30/Java_programming_train.json \
    --eval_data_path ./data/Java_part_programming30/Java_programming_dev.json \
    --continue_train False