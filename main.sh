CUDA_VISIBLE_DEVICES=3 python main.py \
    --do_train True \
    --do_eval True \
    --debug_mode False \
    --learning_rate 1e-4 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --num_train_epochs 100