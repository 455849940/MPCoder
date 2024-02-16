CUDA_VISIBLE_DEVICES=$1 python predict.py \
    --debug_mode False \
    --forwardChoose2 2 \
    --output_dir2 ./stylePrompt_model/stylePrompt_modelB/Short1121/MSAAdapterM55_noCL  \
    --per_device_test_batch_size 4 \
    --user_style_data_path ../data/Java_programming/Java_feature/user_style.json \
    --train_data_path ../data/Java_programming/Java_programming_train.json \
    --eval_data_path ../data/Java_programming/Java_programming_dev.json \
    --test_data_path ../data/Java_programming/java_programming_test_part/Java_programming_test_part$2.json \
    --predict_dirs  ../out_predict/Shot1121_MSAAdapterM55_noCL_part$2.json