#!/bin/bash
set -e  # Exit script immediately if a command fails

model="gnn_baseline"

#model_label="default"
model_label=$1

#use_onehot="true"
use_onehot="false"
#use_onehot=$( [[ "$model" == *"gsp"* || "$model" == *"gnn"* ]] && echo "true" || echo "false" )

#col_embed_model="mpnet"
#col_embed_model="sts"
#col_embed_model="qwen"
col_embed_model=$2

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

batch_sizes=(
    1024 1024 1024 1024 256 1024 1024 512 256 512 1024
)

trainers=(
    "base_trainer" "base_trainer" "base_trainer" "base_trainer"
    "base_trainer" "base_trainer_cls" "base_trainer_cls" "base_trainer_cls"
    "base_trainer_cls" "base_trainer_cls" "base_trainer_cls"
)

eval_metrics=(
    "rmse" "rmse" "rmse" "rmse" "rmse" "accuracy"
    "accuracy" "accuracy" "accuracy" "accuracy" "accuracy"
)

directions=(
    "minimize" "minimize" "minimize" "minimize" "minimize" "maximize"
    "maximize" "maximize" "maximize" "maximize" "maximize"
)

notest=$4

search_label=$5
if [[ "${search_label}" != "" ]]; then
    search="${model}_${search_label}"
else
    search=${model}
fi

label="${col_embed_model}"

for i in $3; do
#for i in "${!dataset_names[@]}"; do
    dataset="${dataset_names[$i]}"
    batch_size="${batch_sizes[$i]}"
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
        trainer.params.data_loader.batch_size=${batch_size} \
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
            trainer.params.data_loader.batch_size=${batch_size} \
            label=${label} \
            search.eval_metric=${eval_metric} \
            search.direction=${direction} \
            seed=$seed \
            test=true
    done
done
