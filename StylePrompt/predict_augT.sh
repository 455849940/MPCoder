CUDA_VISIBLE_DEVICES=1 python predict.py \
    --debug_mode False \
    --learning_rate 1e-5 \
    --per_device_test_batch_size 4 \
    --predict_dirs  ../out_predict/feature_style_promot_result.json
