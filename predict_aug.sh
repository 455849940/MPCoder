CUDA_VISIBLE_DEVICES=4 python predict.py \
    --debug_mode False \
    --learning_rate 1e-4 \
    --per_device_test_batch_size 2 \
    --choose_model_name perfer_Aug\
    --output_dir aug_model \
    --predict_dirs  ./out_predict/aug_result_part_fall.json
