CUDA_VISIBLE_DEVICES=7 python predict.py \
    --debug_mode False \
    --forwardChoose2 1 \
    --output_dir2 ./stylePrompt_model/stylePrompt_modelB/Short1121/MSAAdapterM55_1 \
    --is_predict False \
    --human_eval True \
    --human_uid $1 \
    --human_eval_out_path ../out_predict/humaneval_MPCode_Short1121/prompt2/ \
    --user_style_data_path ../data/Java_programming/Java_feature/user_style.json \
    --user_id2vid_data_path ../data/Java_programming/user_vid2id.json