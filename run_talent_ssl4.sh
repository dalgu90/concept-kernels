#!/bin/bash
set -e  # Exit script immediately if a command fails

#model="ssl_ftt"
#model="ssl_gnn"
model=$1

use_onehot="true"
#use_onehot="false"
#use_onehot=$( [[ "$model" == *"gsp"* || "$model" == *"gnn"*  || "$model" == *"ssl"* ]] && echo "true" || echo "false" )

#col_embed_model="mpnet"
#col_embed_model="sts"
#col_embed_model="qwen"
col_embed_model=$2

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

#phase="pretrain"
#phase="finetune"
phase=$6

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

# 7 8 19 15 102 8 14 95 140 35 20
# 9 12 20 60 102 14 33 95 141 140 140 63
num_transitions=(1 1 2 2 5 1 2 5 5 3 2)
pretrain_max_epochs=(400 100 300 1000 500 200 1000 200 1000 200 400)

for i in $7; do
#for i in "${!dataset_names[@]}"; do
    dataset="${dataset_names[$i]}"
    finetune_trainer="${trainers[$i]}"
    num_transition="${num_transitions[$i]}"
    pretrain_max_epoch="${pretrain_max_epochs[$i]}"
    if [[ "$enctune" == "true" ]]; then
        model2="${model}/${dataset}"
    else
        model2="${model}/default"
    fi
    if [[ "$phase" == "pretrain" ]]; then
        python main.py --config-name ssl \
            dataset=talent/${dataset} \
            model=${model2} \
            phase=pretrain \
            trainer=ssl_pretrain \
            dataset.data_common.use_onehot=$use_onehot \
            dataset.data_common.col_embed_model=${col_embed_model} \
            model.params.ssl_loss_type=${ssl_loss_type} \
            model.params.num_swap_reg=${num_swap_reg} \
            trainer.params.data_augment.name=transition4 \
            trainer.params.data_augment.params.num_transition=$num_transition \
            trainer.params.max_epochs=${pretrain_max_epoch} \
            trainer.params.lr_scheduler.params.num_epochs=${pretrain_max_epoch} \
            label=trans4_${num_swap_reg}_${col_embed_model}
    elif [[ "$phase" == "extract" ]]; then
        python main.py --config-name ssl \
            dataset=talent/${dataset} \
            model=${model2} \
            phase=pretrain \
            trainer=ssl_pretrain \
            dataset.data_common.use_onehot=$use_onehot \
            dataset.data_common.col_embed_model=${col_embed_model} \
            model.params.ssl_loss_type=${ssl_loss_type} \
            model.params.num_swap_reg=${num_swap_reg} \
            trainer.params.data_augment.name=transition4 \
            trainer.params.data_augment.params.num_transition=$num_transition \
            trainer.params.max_epochs=${pretrain_max_epoch} \
            trainer.params.lr_scheduler.params.num_epochs=${pretrain_max_epoch} \
            trainer.params.test_last_ckpt=true \
            trainer.params.test_save_output=true \
            label=trans4_${num_swap_reg}_${col_embed_model} \
            test=true
    else
        for seed in {0..14}; do
            for test in {false,true}; do
                python main.py --config-name ssl \
                    dataset=talent/${dataset} \
                    model=${model2} \
                    phase=finetune \
                    trainer=${finetune_trainer} \
                    dataset.data_common.use_onehot=$use_onehot \
                    dataset.data_common.col_embed_model=${col_embed_model} \
                    model.params.ssl_loss_type=${ssl_loss_type} \
                    model.params.num_swap_reg=${num_swap_reg} \
                    trainer.params.max_epochs=200 \
                    label=trans4_${num_swap_reg}_${col_embed_model} \
                    seed=$seed \
                    test=${test}
            done
        done
    fi
done
