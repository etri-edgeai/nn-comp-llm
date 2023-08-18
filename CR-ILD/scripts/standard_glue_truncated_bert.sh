#!/bin/bash
TEACHER_PATH=./saved/bert_base/ft
STUDENT_PATH=./pretrain/bert_small
OUTPUT_PATH=./saved/standard_glue_truncated_bert
DATA_DIR=./glue_data

SEED="42 43 44"
TASK=${2}


if [ "${1}" == "0" ]; then
    # CR-ILD
    for s in $SEED; do
        CUDA_VISIBLE_DEVICES=0 python run_finetune.py \
            --student_model $STUDENT_PATH \
            --teacher_model $TEACHER_PATH/${TASK} \
            --data_dir $DATA_DIR \
            --task_name ${TASK} \
            --output_dir $OUTPUT_PATH/CRILD/${TASK}/seed${s} \
            --do_lower_case \
            --learning_rate 3e-05 \
            --max_seq_length 128 \
            --train_batch_size 32 \
            --mode TinyBERT-QKV --save_final_model \
            --use_mixup --pkd_initialize \
            --use_ic_att --use_ic_rep --w_ic 0.1 \
            --seed ${s}

        CUDA_VISIBLE_DEVICES=0 python run_finetune.py \
            --student_model $OUTPUT_PATH/CRILD/${TASK}/seed${s} \
            --teacher_model $TEACHER_PATH/${TASK} \
            --data_dir $DATA_DIR \
            --task_name ${TASK} \
            --output_dir $OUTPUT_PATH/CRILD/${TASK}/seed${s}/3e-5 \
            --do_lower_case \
            --learning_rate 3e-05 \
            --num_train_epochs 4 \
            --eval_step 50 \
            --max_seq_length 128 \
            --train_batch_size 16 \
            --mode KD --seed ${s}
            
    done
        
elif [ "${1}" == "1" ]; then
    # TinyBERT
    for s in $SEED; do
        CUDA_VISIBLE_DEVICES=0 python run_finetune.py \
            --student_model $STUDENT_PATH \
            --teacher_model $TEACHER_PATH/${TASK} \
            --data_dir $DATA_DIR \
            --task_name ${TASK} \
            --output_dir $OUTPUT_PATH/TinyBERT/${TASK}/seed${s} \
            --do_lower_case \
            --learning_rate 3e-05 \
            --max_seq_length 128 \
            --train_batch_size 32 \
            --mode TinyBERT --save_final_model \
            --pkd_initialize --hidden_l2l --attention_l2l \
            --seed ${s}

        CUDA_VISIBLE_DEVICES=0 python run_finetune.py \
            --student_model $OUTPUT_PATH/TinyBERT/${TASK}/seed${s} \
            --teacher_model $TEACHER_PATH/${TASK} \
            --data_dir $DATA_DIR \
            --task_name ${TASK} \
            --output_dir $OUTPUT_PATH/TinyBERT/${TASK}/seed${s}/3e-5 \
            --do_lower_case \
            --learning_rate 3e-05 \
            --num_train_epochs 4 \
            --eval_step 50 \
            --max_seq_length 128 \
            --train_batch_size 16 \
            --mode KD --seed ${s}
    done
        
elif [ "${1}" == "2" ]; then
    # BERT-EMD
    for s in $SEED; do
        CUDA_VISIBLE_DEVICES=0 python run_finetune.py \
            --student_model $STUDENT_PATH \
            --teacher_model $TEACHER_PATH/${TASK} \
            --data_dir $DATA_DIR \
            --task_name ${TASK} \
            --output_dir $OUTPUT_PATH/EMD/${TASK}/seed${s} \
            --do_lower_case \
            --learning_rate 3e-05 \
            --max_seq_length 128 \
            --train_batch_size 32 \
            --mode EMD --save_final_model \
            --pkd_initialize --use_att --use_rep \
            --update_weight --seperate --seed ${s}

        CUDA_VISIBLE_DEVICES=0 python run_finetune.py \
            --student_model $OUTPUT_PATH/EMD/${TASK}/seed${s} \
            --teacher_model $TEACHER_PATH/${TASK} \
            --data_dir $DATA_DIR \
            --task_name ${TASK} \
            --output_dir $OUTPUT_PATH/EMD/${TASK}/seed${s}/3e-5 \
            --do_lower_case \
            --learning_rate 3e-05 \
            --num_train_epochs 4 \
            --eval_step 50 \
            --max_seq_length 128 \
            --train_batch_size 16 \
            --mode KD --seed ${s}
    done
        
elif [ "${1}" == "3" ]; then
    # PKD
    for s in $SEED; do
        CUDA_VISIBLE_DEVICES=0 python run_finetune.py \
            --student_model $STUDENT_PATH \
            --teacher_model $TEACHER_PATH/${TASK} \
            --data_dir $DATA_DIR \
            --task_name ${TASK} \
            --output_dir $OUTPUT_PATH/PKD/${TASK}/seed${s} \
            --do_lower_case \
            --learning_rate 3e-05 \
            --max_seq_length 128 \
            --train_batch_size 32 \
            --mode TinyBERT-P --save_final_model \
            --pkd_initialize --hidden_l2l \
            --seed ${s}

        CUDA_VISIBLE_DEVICES=0 python run_finetune.py \
            --student_model $OUTPUT_PATH/PKD/${TASK}/seed${s} \
            --teacher_model $TEACHER_PATH/${TASK} \
            --data_dir $DATA_DIR \
            --task_name ${TASK} \
            --output_dir $OUTPUT_PATH/PKD/${TASK}/seed${s}/3e-5 \
            --do_lower_case \
            --learning_rate 3e-05 \
            --num_train_epochs 4 \
            --eval_step 50 \
            --max_seq_length 128 \
            --train_batch_size 16 \
            --mode KD --seed ${s}
    done

fi
