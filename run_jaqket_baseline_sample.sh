#!/bin/bash -x

DDIR=../data_20200116/

python jaqket_baseline.py  \
  --data_dir   ${DDIR} \
  --model_name_or_path bert-base-japanese-whole-word-masking \
  --task_name jpquiz \
  --output_dir ${OUTDIR} \
  --train_num_options 4 \
  --do_train \
  --do_eval \
  --do_test \
  --per_gpu_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 5 \
  --logging_steps 50 \
  --save_steps 1000

DEV=dev_QBIK.jsonl
TEST=test_QBIK.jsonl
python run_multiple_choice.py  \
  --data_dir   ${DDIR} \
  --dev_fname  ${DEV}  \
  --test_fname ${TEST} \
  --task_name jpquiz \
  --model_name_or_path ${OUTDIR} \
  --eval_num_options 20 \
  --per_gpu_eval_batch_size 8 \
  --do_test \
  --do_eval

DEV=dev_CAPR.jsonl
TEST=test_CAPR.jsonl
python run_multiple_choice.py  \
  --data_dir   ${DDIR} \
  --dev_fname  ${DEV}  \
  --test_fname ${TEST} \
  --task_name jpquiz \
  --model_name_or_path ${OUTDIR} \
  --eval_num_options 20 \
  --per_gpu_eval_batch_size 8 \
  --do_test \
  --do_eval
