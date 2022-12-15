export CUDA_VISIBLE_DEVICES=2

for TRAIN_DIAL in CoQA_SAE CoQA_AAVE CoQA_ChcE CoQA_IndE CoQA_Multi
do
    for EVAL_DIAL in CoQA_ChcE CoQA_IndE
    do
	python3 run_coqa.py --model_type roberta \
		--model_name_or_path SALT-NLP/value-roberta-$TRAIN_DIAL \
		--do_eval \
		--data_dir gold_data/$EVAL_DIAL/ \
		--train_file coqa-train-v1.0.json \
		--predict_file coqa-dev-v1.0.json \
		--learning_rate 3e-5 \
		--num_train_epochs 2 \
		--output_dir ./roberta-$TRAIN_DIAL/ \
		--do_lower_case \
		--per_gpu_train_batch_size 8  \
		--gradient_accumulation_steps 2 \
		--max_grad_norm -1 \
		--weight_decay 0.01 >> ./roberta-$TRAIN_DIAL/in_gold_dial_$EVAL_DIAL
    done
done
