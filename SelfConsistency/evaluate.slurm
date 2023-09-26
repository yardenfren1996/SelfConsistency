#! /bin/sh

#SBATCH --job-name=SC_BERT
#SBATCH --output=logs/out_BERT.txt # redirect stdout
#SBATCH --error=logs/err_BERT.txt # redirect stderr
#SBATCH --partition=studentkillable

python self_consistency.py