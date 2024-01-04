CUDA_VISIBLE_DEVICES=5 python predict.py \
    --debug_mode False \
    --output_dir2 ./stylePrompt_model/stylePrompt_modelB/all_B \
    --per_device_test_batch_size 4 \
    --train_data_path ../data/Java_programming/Java_programming_train.json \
    --eval_data_path ../data/Java_programming/Java_programming_dev.json \
    --test_data_path ../data/Java_programming/java_programming_test_part/Java_programming_test_part4.json \
    --predict_dirs  ../out_predict/style_promot_MSA_all_BqI_part4.json 
