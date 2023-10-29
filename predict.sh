CUDA_VISIBLE_DEVICES=0 python predict.py \
    --debug_mode False \
    --learning_rate 1e-4 \
    --per_device_test_batch_size 1 \
    --output_dir part_model_1001
