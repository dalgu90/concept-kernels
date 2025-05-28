#!/bin/bash
set -e  # Exit script immediately if a command fails

#model="realmlp"
#model="mlp_smooth"
model=$1

#model_label="default"
model_label=$2

#use_onehot="true"
#use_onehot="false"
use_onehot=$( [[ "$model" == *"smooth"* || "$model" == *"gsp"* ]] && echo "true" || echo "false" )
if [[ "$model_label" == *"_te"* ]]; then
    use_onehot="false"
fi

#col_embed_model="mpnet"
#col_embed_model="sts"
#col_embed_model="qwen"
col_embed_model=$3

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
    "realmlp_trainer" "realmlp_trainer" "realmlp_trainer" "realmlp_trainer"
    "realmlp_trainer" "realmlp_trainer_cls" "realmlp_trainer_cls" "realmlp_trainer_cls"
    "realmlp_trainer_cls" "realmlp_trainer_cls" "realmlp_trainer_cls"
)

eval_metrics=(
    "rmse" "rmse" "rmse" "rmse" "rmse" "accuracy"
    "accuracy" "accuracy" "accuracy" "accuracy" "accuracy"
)

directions=(
    "minimize" "minimize" "minimize" "minimize" "minimize" "maximize"
    "maximize" "maximize" "maximize" "maximize" "maximize"
)

notest=$5

search_label=$6
if [[ "${search_label}" != "" ]]; then
    search="${model}_${search_label}"
else
    search=${model}
fi

label="${col_embed_model}"

for i in $4; do
#for i in "${!dataset_names[@]}"; do
    dataset="${dataset_names[$i]}"
    trainer="${trainers[$i]}"
    eval_metric="${eval_metrics[$i]}"
    direction="${directions[$i]}"
    python hparam_search.py \
        dataset=talent/${dataset} \
        model=${model}/${model_label} \
        trainer=${trainer} \
        search=${search} \
        dataset.data_common.use_onehot=${use_onehot} \
        dataset.data_common.col_embed_model=${col_embed_model} \
        label=${label} \
        search.eval_metric=${eval_metric} \
        search.direction=${direction}

    if [[ "$notest" == "true" ]]; then
        exit;
    fi

    for seed in {0..14}; do
        python hparam_search.py \
            dataset=talent/${dataset} \
            model=${model}/${model_label} \
            trainer=${trainer} \
            search=${search} \
            dataset.data_common.use_onehot=$use_onehot \
            dataset.data_common.col_embed_model=${col_embed_model} \
            label=${label} \
            search.eval_metric=${eval_metric} \
            search.direction=${direction} \
            seed=$seed \
            test=true
    done
done
