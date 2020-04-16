#!/bin/bash -x

DDIR=../data/
OUTDIR=output_dir

TRAIN=train_questions.json
DEV=dev1_questions.json
TEST=dev2_questions.json
ENTITY=all_entities.json.gz

python ../JAQKET_baseline/jaqket_baseline.py  \
  --data_dir   ${DDIR} \
  --model_name_or_path bert-base-japanese-whole-word-masking \
  --task_name jaqket \
  --entities_fname ${ENTITY} \
  --train_fname ${TRAIN} \
  --dev_fname   ${DEV} \
  --test_fname  ${TEST} \
  --output_dir ${OUTDIR} \
  --train_num_options 4 \
  --do_train \
  --do_eval \
  --do_test \
  --per_gpu_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 5 \
  --logging_steps 10 \
  --save_steps 1000 \

#--init_global_step 1000 \
#--overwrite_output_dir

python ../JAQKET_baseline/jaqket_baseline.py \
  --data_dir   ${DDIR} \
  --dev_fname  ${DEV}  \
  --test_fname ${TEST} \
  --task_name jaqket \
  --model_name_or_path ${OUTDIR} \
  --eval_num_options 20 \
  --per_gpu_eval_batch_size 4 \
  --do_test \
  --do_eval
