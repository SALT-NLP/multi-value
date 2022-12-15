export CUDA_VISIBLE_DEVICES=1

# python3 run_coqa.py --model_type roberta \
    #         --model_name_or_path roberta-base \
    #         --do_train \
    #         --do_eval \
    #         --data_dir data/ \
    #         --train_file coqa-train-v1.0.json \
    #         --predict_file coqa-dev-v1.0.json \
    #         --learning_rate 3e-5 \
    #         --num_train_epochs 2 \
    #         --output_dir roberta-output/ \
    #         --per_gpu_train_batch_size 8  \
    # 	--gradient_accumulation_steps 2 \
    #         --max_grad_norm -1 \
    #         --weight_decay 0.01

#CoQA_AAVE CoQA_AppE CoQA_ChcE CoQA_CollSgE CoQA_IndE CoQA_Multi
for DIALECT_DATA in CoQA_CollSgE CoQA_IndE
do
    python3 run_coqa.py --model_type roberta \
            --model_name_or_path roberta-base \
            --do_train \
            --do_eval \
            --data_dir eval_data/$DIALECT_DATA/ \
            --train_file coqa-train-v1.0.json \
            --predict_file coqa-dev-v1.0.json \
            --learning_rate 3e-5 \
            --num_train_epochs 2 \
            --output_dir /data/wheld3/roberta-$DIALECT_DATA/ \
            --do_lower_case \
            --per_gpu_train_batch_size 8  \
	    --gradient_accumulation_steps 2 \
            --max_grad_norm -1 \
            --weight_decay 0.01
done
