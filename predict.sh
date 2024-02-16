CUDA_VISIBLE_DEVICES=6 python predict.py \
    --debug_mode False \
    --per_device_test_batch_size 4 \
    --output_dir style_model/Long50 \
    --train_data_path ./data/Java_part_programmingLong50/Java_programming_train.json \
    --eval_data_path ./data/Java_part_programmingLong50/Java_programming_dev.json \
    --test_data_path ./data/Java_part_programmingLong50/Java_programming_test.json \
    --predict_dirs  ./out_predict/soft_prompt_Long50.json \
    --human_eval_out_path ./humaneval_data/humeval_result_style_model_data.jsonl