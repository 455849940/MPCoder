CUDA_VISIBLE_DEVICES=1 python predict.py \
    --debug_mode False \
    --learning_rate 1e-5 \
    --per_device_test_batch_size 3 \
    --train_data_path ../data/Java_part_programming50/Java_programming_train.json \
    --eval_data_path ../data/Java_part_programming50/Java_programming_dev.json \
    --test_data_path ../data/Java_part_programming50/Java_programming_test.json \
    --predict_dirs  ../out_predict/feature_style_promot_result_50_e5.json \
