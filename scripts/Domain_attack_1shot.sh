
seed=42
N=1 # ontonotes: 4, wnut: 3, gum: 2, conll: 1
K=1 # k-shot 
device=0 
experiment_name=conll_1shot

# pretraining on source domain 
python main.py \
    --data_path=episode-data/ \
    --types_path=episode-data/entity_types_domain.json \
    --N=${N} \
    --K=${K} \
    --tagging_scheme=BIOES \
    --bert_model=bert-base-uncased \
    --sample_size=5 \
    --max_seq_len=128 \
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
    --choice_number=10 \
    --is_attack=attack \
    --ignore_eval_test


# finetuning & evaluate on target domain 
python main.py \
    --data_path=episode-data/ \
    --types_path=episode-data/entity_types_domain.json \
    --N=${N} \
    --K=${K} \
    --tagging_scheme=BIOES \
    --bert_model=bert-base-uncased \
    --sample_size=5 \
    --max_seq_len=128 \
    --project_type_embedding=True \
    --type_embedding_size=128 \
    --mode=${mode} \
    --data_model=${data_model} \
    --gpu_device=${device} \
    --seed=${seed} \
    --name=${experiment_name} \
    --batch_size=1 \
    --lr=3e-5 \
    --max_train_steps=1000 \
    --eval_every_train_steps=100 \
    --lr_finetune=3e-5 \
    --max_finetune_steps=50 \
    --ignore_eval_test \
    --choice_number=10 \
    --is_attack=attack \
    --test_only 