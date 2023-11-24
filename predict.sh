CUDA_VISIBLE_DEVICES=1 python predict.py \
    --debug_mode False \
    --learning_rate 1e-4 \
    --per_device_test_batch_size 6 \
    --output_dir style_model \
    --train_data_path ./data/Java_programming/Java_programming_train.json \
    --eval_data_path ./data/Java_programming/Java_programming_dev.json \
    --test_data_path ./data/Java_programming/Java_programming_test.json \
    --predict_dirs  ./out_predict/result_style_model.json \
    --human_eval_out_path ./humaneval_data/humeval_result_style_model_data.jsonl