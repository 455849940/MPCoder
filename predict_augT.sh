CUDA_VISIBLE_DEVICES=5 python predict.py \
    --debug_mode False \
    --learning_rate 1e-4 \
    --per_device_test_batch_size 1 \
    --choose_model_name perfer_AugT\
    --output_dir augT_model_linear \
    --predict_dirs  ./out_predict/augT_result_part50.json
