#!/bin/bash
set -e  # Exit script immediately if a command fails

#model="realmlp"
#model="mlp_smooth"
#model="mlp_gsp"
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

label="${col_embed_model}"

for i in $4; do
#for i in "${!dataset_names[@]}"; do
    dataset="${dataset_names[$i]}"
    trainer="${trainers[$i]}"
    for seed in {0..14}; do
        for test in {false,true}; do
            python main.py \
                dataset=talent/${dataset} \
                model=${model}/${model_label} \
                trainer=${trainer} \
                dataset.data_common.use_onehot=$use_onehot \
                dataset.data_common.col_embed_model=${col_embed_model} \
                label=${label} \
                seed=$seed \
                test=${test}
        done
        #python main.py \
            #dataset=talent/${dataset} \
            #model=${model}/${model_label} \
            #trainer=${trainer} \
            #dataset.data_common.use_onehot=$use_onehot \
            #dataset.data_common.col_embed_model=${col_embed_model} \
            #label=${label} \
            #seed=0 \
            #test=true
    done
done
