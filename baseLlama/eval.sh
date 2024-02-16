CUDA_VISIBLE_DEVICES=3 python predict_eval_test_data.py \
    --per_device_test_batch_size 3 \
    --train_data_path /home/develop/dzl/PreferCodeLlama/data/Java_part_programmingLong50/Java_programming_train.json \
    --eval_data_path /home/develop/dzl/PreferCodeLlama/data/Java_part_programmingLong50/Java_programming_dev.json \
    --test_data_path /home/develop/dzl/PreferCodeLlama/data/Java_part_programmingLong50/Java_programming_test.json\