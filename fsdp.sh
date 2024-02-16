export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=1,2,3,5 torchrun --master_port=65530 --nproc_per_node 4 main.py \
    --do_train True \
    --do_eval True \
    --debug_mode False\
    --learning_rate 1e-4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 30 \
    --output_dir style_model/short1121 \
    --train_data_path ./data/Java_programming/Java_programming_train.json \
    --eval_data_path ./data/Java_programming/Java_programming_dev.json \
    --continue_train False