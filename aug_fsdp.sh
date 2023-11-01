export NCCL_P2P_DISABLE=1
CUDA_VISIBLE_DEVICES=1,2,3,4 torchrun --master_port=65531 --nproc_per_node 4 main.py \
    --do_train True \
    --do_eval True \
    --debug_mode False\
    --learning_rate 1e-3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --num_train_epochs 50 \
    --choose_model_name perfer_Aug\
    --output_dir aug_model