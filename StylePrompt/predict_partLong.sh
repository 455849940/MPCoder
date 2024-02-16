CUDA_VISIBLE_DEVICES=$1 python predict.py \
    --debug_mode False \
    --forwardChoose2 1\
    --output_dir2 ./stylePrompt_model/stylePrompt_modelB/Long50/MPCode_NCL  \
    --per_device_test_batch_size 4 \
    --user_style_data_path ../data/Java_part_programmingLong50/user_style.json \
    --train_data_path ../data/Java_part_programmingLong50/Java_programming_train.json \
    --eval_data_path ../data/Java_part_programmingLong50/Java_programming_dev.json \
    --test_data_path ../data/Java_part_programmingLong50/java_programming_test_part/Java_programming_test_part$2.json \
    --predict_dirs  ../out_predict/Long50_MPCode_NCL_part$2.json