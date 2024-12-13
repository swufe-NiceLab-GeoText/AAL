
seed=42
Ns=(1 2 3 4 5 6 7)
K=1
device=0
data_model=Snips
experiment_name=we_1shot
mode=Snips
# pretraining on source snips
for N in ${Ns[@]}; do
python main.py \
    --data_path=episode-data/ \
    --types_path=episode-data/entity_types_snips.json \
    --N=${N} \
    --K=${K} \
    --tagging_scheme=BIOES \
    --mode=${mode} \
    --data_model=${data_model} \
    --bert_model=bert-base-uncased \
    --max_seq_len=128 \
    --sample_size=5 \
    --project_type_embedding=True \
    --type_embedding_size=128 \
    --gpu_device=${device} \
    --seed=${seed} \
    --name=${experiment_name} \
    --batch_size=1 \
    --lr=3e-5 \
    --max_train_steps=1000 \
    --eval_every_train_steps=100 \
    --lr_finetune=3e-5 \
    --max_finetune_steps=50 \
    --ignore_eval_test


# finetuning & evaluate on target domain 
python main.py \
    --data_path=episode-data/ \
    --types_path=episode-data/entity_types_snips.json \
    --N=${N} \
    --K=${K} \
    --tagging_scheme=BIOES \
    --mode=${mode} \
    --data_model=${data_model} \
    --bert_model=bert-base-uncased \
    --sample_size=5 \
    --max_seq_len=128 \
    --project_type_embedding=True \
    --type_embedding_size=128 \
    --gpu_device=${device} \
    --seed=${seed} \
    --name=${experiment_name} \
    --batch_size=1 \
    --lr=3e-5 \
    --max_train_steps=1000 \
    --eval_every_train_steps=100 \
    --lr_finetune=3e-5 \
    --max_finetune_steps=50 \
    --choice_number=3 \
    --ignore_eval_test \
    --test_only

done