#!/bin/bash

HF_TOKEN='YOUR_ACCESS_TOKEN'
INPUT_FILE='comments_sample.csv'

MODEL_IDS=('facebook/opt-1.3b' 'meta-llama/Llama-3.1-8B-Instruct')

PROMPTS="Is the following post Misogynistic? (yes/no). Post:;Classify this text as Misogynistic or Non-Misogynistic. Text:"

filename=${INPUT_FILE%".csv"}
for model_id in "${MODEL_IDS[@]}";
do
    model_name="$(echo "$model_id" | cut -d "/" -f 2)"
    python get_video_llms.py \
        --hf_access_token "$HF_TOKEN" \
        --model_id "$model_id" \
        --prompts "$PROMPTS" \
        --input_file "$INPUT_FILE" \
        --text_col text &> "$filename"_"$model_name".log

done
    