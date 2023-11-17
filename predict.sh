CUDA_VISIBLE_DEVICES=5 python predict.py \
    --debug_mode False \
    --learning_rate 1e-4 \
    --per_device_test_batch_size 2 \
    --output_dir style_model \
    --predict_dirs  ./out_predict/result_style_model_linear_50_batch4.json \
    --human_eval_out_path ./humaneval_data/humeval_result_style_model_data.jsonl