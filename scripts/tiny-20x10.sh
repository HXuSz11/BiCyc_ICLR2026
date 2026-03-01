#!/bin/bash
for SEED in 0 1 2 3 4
do
  for VAL in 1
  do
    CUDA_VISIBLE_DEVICES=3 python src/main_incremental.py \
    --approach bicyc \
    --datasets tiny --num-tasks 20 --nc-first-task 10 \
    --nepochs 200 --batch-size 256 \
    --criterion ce --classifier bayes \
    --S 64 --lamb 5 --lr 0.1 --weight-decay 5e-4 --normalize --rotation \
    --distillation projected_bi --lambda-bi 5 --lambda-cycle 2 --pair-preserve --lambda-pair 0.05 \
    --exp-name 20x10_seed${SEED}/ --results-path ../results_bicyc_ICLR_camera_ready/
  done
done
