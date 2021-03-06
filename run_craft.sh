#!/bin/bash

mode=$1  # all, train (only), baseline, langemb
nformula=$2  # number of formulas in training
runid=$3  # index for the run
lr=$4  # learning rate, set default 0.001 for craft
rnnsize=$5  # rnn size, default 64
rnndepth=$6  # rnn depth, default 1

env_flag="--env_name Craft --num_steps 15"
env_flag_train="--env_name Craft --num_steps 15"
gen_train_formula="--gen_formula_only --num_train_ltls $nformula"
gen_test_formula="--gen_formula_only --num_test_ltls 100"
baseline_flag=""
trans_flag=""

home="/home/"

train_path="--formula_pickle ${home}data/formula_Craft_${nformula}_${runid}.pickle"
test_path_0="${home}data/formula_Craft_test_in_${nformula}_${runid}.pickle"
test_path_1="${home}data/formula_Craft_test_out_${nformula}_${runid}.pickle"
test_path_2="${home}data/formula_Craft_test_out15_${nformula}_${runid}.pickle"
test_path_3="${home}data/formula_Craft_test_out20_${nformula}_${runid}.pickle"

model_path="${home}models/a2c/Craft_${nformula}_${runid}/"
model_name="Craft_${nformula}_${runid}/model"


# set a different model path for baselines
if [ $mode = "baseline" ]; then
    model_path="${home}models/a2c/Craft_base_${nformula}_${runid}/"
    model_name="Craft_base_${nformula}_${runid}/model"
    baseline_flag="--baseline"
fi
if [ $mode = "langemb" ]; then
    model_path="${home}models/a2c/Craft_lang_${nformula}_${runid}/"
    model_name="Craft_lang_${nformula}_${runid}/model"
    baseline_flag="--baseline --lang_emb"
fi
if [ $mode = "notime" ]; then
    baseline_flag="--no_time"
fi


# generate data for 'all' mode
if [ $mode = "all" ]; then
    # generate training data
    python ${home}main.py $env_flag $gen_train_formula $train_path \
        --test_formula_pickle_1 $test_path_1 \
        --num_test_ltls 100

    # generate testing data
    python ${home}main.py $env_flag $gen_test_formula $train_path \
        --test_formula_pickle_1 $test_path_0 \
        --num_test_ltls 100 --test_in_domain

    python ${home}main.py $env_flag $gen_test_formula $train_path \
        --test_formula_pickle_1 $test_path_2 \
        --num_test_ltls 100 --test_out_domain \
        --min_symbol_len 10 --max_symbol_len 15

    python ${home}main.py $env_flag $gen_test_formula $train_path \
        --test_formula_pickle_1 $test_path_3 \
        --num_test_ltls 100 --test_out_domain \
        --test_out_domain --min_symbol_len 15 --max_symbol_len 20
fi


# make the directory for the models
mkdir $model_path

# train the model
python ${home}main.py $env_flag_train $trans_flag --algo a2c \
    --log_dir /tmp/ltl-rl-craft-${nformula}/ \
    --prefix_reward_decay 0.6 \
    --use_gae --entropy_coef 0.1 --gamma 0.9 --train \
    --load_formula_pickle $train_path --num_train_ltls $nformula \
    --save_model_name $model_name $baseline_flag \
    --num_processes 15 --num_epochs 1000 \
    --test_formula_pickle_1 $test_path_0 \
    --test_formula_pickle_2 $test_path_1 \
    --test_formula_pickle_3 $test_path_2 \
    --test_formula_pickle_4 $test_path_3 \
    --num_env_steps 450 --log_interval 100 \
    --load_eval_train \
    --image_emb_size 64 \
    --rnn_size $rnnsize \
    --rnn_depth $rnndepth \
    --output_state_size 32 \
    --lr $lr --alpha 0.9 \
    --use_lr_scheduler \
    --lr_scheduled_update 250
