CUDA_VISIBLE_DEVICES=1 python predict.py \
    --debug_mode False \
    --output_dir2 ./stylePrompt_model/stylePrompt_modelB/part50_B \
    --per_device_test_batch_size 4 \
    --train_data_path ../data/Java_part_programming50/Java_programming_train.json \
    --eval_data_path ../data/Java_part_programming50/Java_programming_dev.json \
    --test_data_path ../data/Java_part_programming50/Java_programming_test.json \
    --predict_dirs  ../out_predict/style_promot_MSA_part50_BqI.json 
