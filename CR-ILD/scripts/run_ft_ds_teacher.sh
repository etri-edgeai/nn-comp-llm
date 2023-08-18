#!/bin/bash
export TEACHER_PATH=./pretrain/bert_base
export STUDENT_PATH=./pretrain/bert_small
export TEACHER_OUT=./saved/bert_base/ds
export STUDENT_OUT=./saved/bert_small
export DATA_PATH=./glue_data
task_lst="CoLA MRPC RTE STS-B"

for t in $task_lst
do
    CUDA_VISIBLE_DEVICES=0 python run_finetune.py \
        --student_model $TEACHER_PATH \
        --teacher_model $TEACHER_OUT/$t \
        --data_dir $DATA_PATH \
        --task_name $t \
        --output_dir $TEACHER_OUT/$t \
        --do_lower_case \
        --num_train_epochs 20 \
        --learning_rate 2e-05 \
        --eval_step 100 \
        --max_seq_length 128 \
        --train_batch_size 16 \
        --mode FT --save_model \
        --downsample
done
