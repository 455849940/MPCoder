CUDA_VISIBLE_DEVICES=5 python predict.py \
    --debug_mode False \
    --per_device_test_batch_size 4 \
    --choose_model_name perfer_AugT\
    --output_dir augT_model_linear/Short1121_noCL \
    --train_data_path ./data/Java_programming/Java_programming_train.json \
    --eval_data_path ./data/Java_programming/Java_programming_dev.json \
    --test_data_path ./data/Java_programming/Java_programming_test.json \
    --predict_dirs  ./out_predict/style_Adapter_Long50_55.json \
    --human_eval_out_path ./out_predict/humeval_result_aug_model_3.jsonl
