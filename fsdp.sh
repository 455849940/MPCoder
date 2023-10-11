CUDA_VISIBLE_DEVICES=3,5 torchrun --nproc_per_node 2 main.py \
    --do_train True \
    --do_eval True \
    --debug_mode True \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 100