export CUDA_VISIBLE_DEVICES=3

#coqa CoQA_AAVE CoQA_AppE CoQA_ChcE CoQA_CollSgE CoQA_IndE
# CoQA_IndE
for TRAIN_DIAL in coqa
do
    for EVAL_DIAL in CoQA_SAE CoQA_AAVE CoQA_AppE CoQA_ChcE CoQA_CollSgE CoQA_IndE
    do
	python3 run_coqa.py --model_type bert \
		--model_name_or_path /data/wheld3/bert-$TRAIN_DIAL-adapter/ \
		--train_adapter \
		--do_eval \
		--data_dir eval_data/$EVAL_DIAL/ \
		--train_file coqa-train-v1.0.json \
		--predict_file coqa-dev-v1.0.json \
		--learning_rate 3e-5 \
		--num_train_epochs 2 \
		--output_dir /data/wheld3/bert-$TRAIN_DIAL-adapter/ \
		--do_lower_case \
		--tada_adapter WillHeld/pfadapter-bert-base-uncased-tada-value-small \
		--per_gpu_train_batch_size 8  \
		--gradient_accumulation_steps 2 \
		--max_grad_norm -1 \
		--weight_decay 0.01 >> /data/wheld3/bert-$TRAIN_DIAL-adapter/in_dial_$EVAL_DIAL
    done
done
