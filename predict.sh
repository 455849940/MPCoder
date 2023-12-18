CUDA_VISIBLE_DEVICES=7 python predict.py \
    --debug_mode False \
    --learning_rate 1e-5 \
    --per_device_test_batch_size 3 \
    --output_dir style_model \
    --train_data_path ./data/Java_part_programming50/Java_programming_train.json \
    --eval_data_path ./data/Java_part_programming50/Java_programming_dev.json \
    --test_data_path ./data/Java_part_programming50/Java_programming_test.json \
    --predict_dirs  ./out_predict/result_style_model_new_50_e5.json \
    --human_eval_out_path ./humaneval_data/humeval_result_style_model_data.jsonl