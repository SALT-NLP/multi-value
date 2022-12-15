export CUDA_VISIBLE_DEVICES=1

python3 run_coqa.py --model_type roberta-base \
                   --model_name_or_path /data/wheld3/roberta-coqa \
                   --do_eval \
                   --data_dir data/ \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 2 \
                   --output_dir roberta-output/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 8  \
		   --gradient_accumulation_steps 2 \
                   --max_grad_norm -1 \
                   --weight_decay 0.01 >> roberta-output/sae

python3 run_coqa.py --model_type roberta-base \
                   --model_name_or_path /data/wheld3/roberta-coqa \
                   --do_eval \
                   --data_dir eval_data/CoQA_AAVE/ \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 2 \
                   --output_dir roberta-output/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 8  \
		   --gradient_accumulation_steps 2 \
                   --max_grad_norm -1 \
                   --weight_decay 0.01 >> roberta-output/aave

python3 run_coqa.py --model_type roberta-base \
                   --model_name_or_path /data/wheld3/roberta-coqa \
                   --do_eval \
                   --data_dir eval_data/CoQA_ChcE/ \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 2 \
                   --output_dir roberta-output/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 8  \
		   --gradient_accumulation_steps 2 \
                   --max_grad_norm -1 \
                   --weight_decay 0.01 >> roberta-output/chce

python3 run_coqa.py --model_type roberta-base \
                   --model_name_or_path /data/wheld3/roberta-coqa \
                   --do_eval \
                   --data_dir eval_data/CoQA_AppE/ \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 2 \
                   --output_dir roberta-output/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 8  \
		   --gradient_accumulation_steps 2 \
                   --max_grad_norm -1 \
                   --weight_decay 0.01 >> roberta-output/apche

python3 run_coqa.py --model_type roberta-base \
                   --model_name_or_path /data/wheld3/roberta-coqa \
                   --do_eval \
                   --data_dir eval_data/CoQA_CollSgE/ \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 2 \
                   --output_dir roberta-output/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 8  \
		   --gradient_accumulation_steps 2 \
                   --max_grad_norm -1 \
                   --weight_decay 0.01 >> roberta-output/sge

python3 run_coqa.py --model_type roberta-base \
                   --model_name_or_path /data/wheld3/roberta-coqa \
                   --do_eval \
                   --data_dir eval_data/CoQA_IndE/ \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 2 \
                   --output_dir roberta-output/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 8  \
		   --gradient_accumulation_steps 2 \
                   --max_grad_norm -1 \
                   --weight_decay 0.01 >> roberta-output/inde

python3 run_coqa.py --model_type roberta-base \
                   --model_name_or_path /data/wheld3/roberta-coqa \
                   --do_eval \
                   --data_dir eval_data/CoQA_Multi/ \
                   --predict_file coqa-dev-v1.0.json \
                   --learning_rate 3e-5 \
                   --num_train_epochs 2 \
                   --output_dir roberta-output/ \
                   --do_lower_case \
                   --per_gpu_train_batch_size 8  \
		   --gradient_accumulation_steps 2 \
                   --max_grad_norm -1 \
                   --weight_decay 0.01 >> roberta-output/multi
