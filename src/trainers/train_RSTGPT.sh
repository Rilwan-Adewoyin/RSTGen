#!/bin/bash

# SLURM SUBMIT SCRIPT
#SBATCH -p gpu               # submit to gpu partition
#SBATCH -J rstGPT       # JOB NAME
#SBATCH -o output.txt      # Redirection Print Out File
#SBATCH --nodes=3
#SBATCH --gres=gpu:3
#SBATCH --ntasks-per-node=3
#SBATCH --mem=0
#SBATCH --time=24:00:00

srun python3 trainers/train_RSTGPT.py --batch_size 16 --version 1 --workers 12 --num_nodes 3 --gpus 9 --tag "RSTGPT with aligned attention and regularisation" --max_len_utt 270 --max_len_rst 36 --max_len_key_phrase 64 --rst_tree_aligned_attention 1 --rst_segment_method segbot
