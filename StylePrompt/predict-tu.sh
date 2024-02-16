CUDA_VISIBLE_DEVICES=6 python predict_tu.py \
    --debug_mode False \
    --forwardChoose2 3 \
    --output_dir2 ./stylePrompt_model/stylePrompt_modelB/Long50/MPCode_NCL \
    --per_device_test_batch_size 1 \
    --user_style_data_path ../data/Java_part_programmingLong50/user_style.json \
    --train_data_path ../data/Java_part_programmingLong50/Java_programming_train.json \
    --eval_data_path ../data/Java_part_programmingLong50/Java_programming_dev.json \
    --test_data_path ../data/Java_part_programmingLong50/Java_programming_test.json \
    --predict_dirs  ../out_predict/Long50_MPCode_NCL_tu.json 
