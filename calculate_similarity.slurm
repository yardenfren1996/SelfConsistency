#! /bin/sh
#SBATCH --job-name=sim
#SBATCH --output=logs/sim.out
#SBATCH --error=logs/sim.err
#SBATCH --partition=studentbatch

python simmilarity.py --model "noun_ngram" --data "results/pegasus_data_scores.pkl" --results_filename "pegasus_data_scores"
