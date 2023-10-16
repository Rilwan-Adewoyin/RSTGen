#!/bin/bash

#SBATCH -J rstGPT       # JOB NAME

#SBATCH -o output.txt      # Redirection Print Out File

#SBATCH -p gpu               # submit to gpu partition

#SBATCH --nodes=1 

#SBATCH --gres=gpu:3

#SBATCH --mem=0

#SBATCH -c 8

#SBATCH --time=36:00:00

srun python3 train_RSTGPT_arggen_dyploc1.py --batch_size 16 --version 21 --finetune_version 2 --precision 16 --mode finetune --workers 8 --gpus 3 --tag debugging --max_len_utt 270 --max_len_rst 64 --max_len_title 40 --max_len_claim 20 --max_len_key_phrase 64 --tag "ArgGen. Normal Loss" --rst_tree_aligned_attention 0 --rst_segment_method segbot
srun python3 train_RSTGPT_arggen_dyploc2.py --batch_size 9 --version 22 --finetune_version 2 --precision 16 --mode finetune --workers 8 --gpus 3 --tag debugging --max_len_utt 270 --max_len_rst 64 --max_len_title 40 --max_len_claim 20 --max_len_key_phrase 64 --tag "ArgGen. With regularisation loss" --rst_tree_aligned_attention 0 --rst_segment_method segbot
srun python3 train_RSTGPT_arggen_pair1.py --batch_size 16 --version 21 --finetune_version 2 --precision 16 --mode finetune --workers 8 --gpus 3 --tag debugging --max_len_utt 270 --max_len_rst 64 --max_len_title 40 --max_len_key_phrase 64 --tag "ArgGen. Normal Loss" --rst_tree_aligned_attention 0 --rst_segment_method segbot
srun python3 train_RSTGPT_arggen_pair2.py --batch_size 9 --version 22 --finetune_version 2 --precision 16 --mode finetune --workers 8 --gpus 3 --tag debugging --max_len_utt 270 --max_len_rst 64 --max_len_title 40 --max_len_key_phrase 64 --tag "ArgGen. With regularisation loss" --rst_tree_aligned_attention 0 --rst_segment_method segbot