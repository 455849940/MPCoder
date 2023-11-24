CUDA_VISIBLE_DEVICES=5 python predict.py \
    --debug_mode False \
    --learning_rate 1e-4 \
    --per_device_test_batch_size 2 \
    --choose_model_name perfer_Aug\
    --output_dir aug_model_linear \
    --predict_dirs  ./out_predict/aug_result_linear_50_batch4.json \
    --human_eval_out_path ./out_predict/humeval_result_aug_model_3.jsonl
