export TASK_NAME=sst2
export PYTHONHASHSEED=1234
python run_glue.py \
  --model_name_or_path roberta-base \
  --task_name $TASK_NAME \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir results/$TASK_NAME-roberta_base \
  --dialect "aave" \
  --morphosyntax \
  --do_train