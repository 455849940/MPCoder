export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=3,5 torchrun --master_port=65535 --nproc_per_node 2 run.py \
    --do_train_first False \
    --do_train_second True \
    --do_eval True \
    --debug_mode False \
    --learning_rate 1e-5 \
    --per_device_feature_train_batch_size 1\
    --per_device_feature_dev_batch_size 3 \
    --num_feature_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 30 \
    --train_data_path ../data/Java_part_programming5/Java_programming_train.json \
    --eval_data_path ../data/Java_part_programming5/Java_programming_dev.json \
    --continue_train False \
    --enable_contrast True \
    --alpha 0.5