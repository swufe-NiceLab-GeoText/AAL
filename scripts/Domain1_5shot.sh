
seed=42
N=1 # ontonotes: 4, wnut: 3, gum: 2, conll: 1
K=5 # k-shot
device=0 
data_model=Domain
experiment_name=conll_5shot
mode=Domain


# pretraining on source domain 
python main.py \
    --data_path=episode-data/ \
    --types_path=episode-data/entity_types_domain.json \
    --N=${N} \
    --K=${K} \
    --tagging_scheme=BIOES \
    --bert_model=bert-base-uncased \
    --max_seq_len=128 \
    --sample_size=5 \
    --project_type_embedding=True \
    --type_embedding_size=128 \
    --gpu_device=${device} \
    --mode=${mode} \
    --data_model=${data_model} \
    --seed=${seed} \
    --name=${experiment_name} \
    --batch_size=1 \
    --lr=3e-5 \
    --max_train_steps=1000 \
    --eval_every_train_steps=100 \
    --lr_finetune=3e-5 \
    --max_finetune_steps=50 \
    --choice_number=5 \
    --ignore_eval_test


# finetuning & evaluate on target domain 
python main.py \
    --data_path=episode-data/ \
    --types_path=episode-data/entity_types_domain.json \
    --N=${N} \
    --K=${K} \
    --tagging_scheme=BIOES \
    --bert_model=bert-base-uncased \
    --max_seq_len=128 \
    --project_type_embedding=True \
    --type_embedding_size=128 \
    --sample_size=5 \
    --gpu_device=${device} \
    --mode=${mode} \
    --data_model=${data_model} \
    --seed=${seed} \
    --name=${experiment_name} \
    --batch_size=1 \
    --lr=3e-5 \
    --max_train_steps=1000 \
    --eval_every_train_steps=100 \
    --lr_finetune=3e-5 \
    --max_finetune_steps=50 \
    --ignore_eval_test \
    --choice_number=5 \
    --test_only 