#!/bin/bash
set -e  # Exit script immediately if a command fails

#model="ssl_ftt"
#model="ssl_gnn"
model=$1

use_onehot="true"
#use_onehot="false"
#use_onehot=$( [[ "$model" == *"gsp"* || "$model" == *"gnn"* ]] && echo "true" || echo "false" )

#col_embed_model="mpnet"
#col_embed_model="sts"
#col_embed_model="qwen"
col_embed_model=$2

#num_swap_reg="no"
#num_swap_reg="fixcorr"
#num_swap_reg="corr"
#num_swap_reg="fixsign"
num_swap_reg=$3

#ssl_loss_type="SCL"
#ssl_loss_type="InfoNCE"
ssl_loss_type=$4

#enctune="true"
#enctune="false"
enctune=$5

dataset_names=(
    "Abalone_reg"
    "Diamonds"
    "Parkinsons_Telemonitoring"
    "archive_r56_Portuguese"
    "communities_and_crime"
    "Bank_Customer_Churn_Dataset"
    "statlog"
    "taiwanese_bankruptcy_prediction"
    "ASP-POTASSCO-classification"
    "internet_usage"
    "predict_students_dropout_and_academic_success"
)

trainers=(
    "ssl_finetune" "ssl_finetune" "ssl_finetune" "ssl_finetune"
    "ssl_finetune" "ssl_finetune_cls" "ssl_finetune_cls" "ssl_finetune_cls"
    "ssl_finetune_cls" "ssl_finetune_cls" "ssl_finetune_cls"
)

eval_metrics=(
    "rmse" "rmse" "rmse" "rmse" "rmse" "accuracy"
    "accuracy" "accuracy" "accuracy" "accuracy" "accuracy"
)

directions=(
    "minimize" "minimize" "minimize" "minimize" "minimize" "maximize"
    "maximize" "maximize" "maximize" "maximize" "maximize"
)

notest=$7

for i in $6; do
#for i in "${!dataset_names[@]}"; do
    dataset="${dataset_names[$i]}"
    trainer="${trainers[$i]}"
    eval_metric="${eval_metrics[$i]}"
    direction="${directions[$i]}"
    if [[ "$enctune" == "true" ]]; then
        model2="${model}/${dataset}"
    else
        model2="${model}/default"
    fi
    python hparam_search.py --config-name ssl_hparam \
        dataset=talent/${dataset} \
        model=${model2} \
        trainer=${trainer} \
        search=${model} \
        phase=finetune \
        dataset.data_common.use_onehot=$use_onehot \
        dataset.data_common.col_embed_model=${col_embed_model} \
        model.params.ssl_loss_type=${ssl_loss_type} \
        label=trans4_${num_swap_reg}_${col_embed_model} \
        search.eval_metric=${eval_metric} \
        search.direction=${direction}

    if [[ "$notest" == "true" ]]; then
        exit;
    fi

    for seed in {0..14}; do
        python hparam_search.py --config-name ssl_hparam \
            dataset=talent/${dataset} \
            model=${model2} \
            trainer=${trainer} \
            search=${model} \
            phase=finetune \
            dataset.data_common.use_onehot=$use_onehot \
            dataset.data_common.col_embed_model=${col_embed_model} \
            model.params.ssl_loss_type=${ssl_loss_type} \
            search.eval_metric=${eval_metric} \
            search.direction=${direction} \
            label=trans4_${num_swap_reg}_${col_embed_model} \
            seed=$seed \
            test=true
    done
done
