CUDA_VISIBLE_DEVICES=3 python predict_eval_test_data.py \
    --per_device_test_batch_size 2 \
    --train_data_path /home/develop/dzl/PreferCodeLlama/data/Java_part_programming50/Java_programming_train.json \
    --eval_data_path /home/develop/dzl/PreferCodeLlama/data/Java_part_programming50/Java_programming_dev.json