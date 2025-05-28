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
    1024 1024 1024 1024 256 1024 1024 512 256 1024 1024
)

trainers=(
    "base_trainer"
    "base_trainer"
    "base_trainer"
    "base_trainer"
    "base_trainer"
    "base_trainer_cls"
    "base_trainer_cls"
    "base_trainer_cls"
    "base_trainer_cls"
    "base_trainer_cls"
    "base_trainer_cls"
)

label="${col_embed_model}"

for i in $3; do
#for i in "${!dataset_names[@]}"; do
    dataset="${dataset_names[$i]}"
    batch_size="${batch_sizes[$i]}"
    trainer="${trainers[$i]}"
    for seed in {0..14}; do
        for test in {false,true}; do
            python main.py \
                dataset=talent/${dataset} \
                model=${model}/${model_label} \
                trainer=${trainer} \
                trainer.params.data_loader.batch_size=${batch_size} \
                trainer.params.max_epochs=200 \
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
            #trainer.params.data_loader.batch_size=${batch_size} \
            #trainer.params.max_epochs=200 \
            #dataset.data_common.use_onehot=$use_onehot \
            #dataset.data_common.col_embed_model=${col_embed_model} \
            #label=${label} \
            #seed=0 \
            #test=true
    done
done
