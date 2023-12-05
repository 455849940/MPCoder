CUDA_VISIBLE_DEVICES=5 python predict_eval_test_data.py \
    --per_device_test_batch_size 4 \
    --train_data_path /home/develop/dzl/PreferCodeLlama/data/Java_programming/Java_programming_train.json \
    --eval_data_path /home/develop/dzl/PreferCodeLlama/data/Java_programming/Java_programming_dev.json