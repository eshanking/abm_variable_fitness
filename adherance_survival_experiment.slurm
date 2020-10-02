#!/bin/bash
#SBATCH -N 1 
#SBATCH -c 1 
#SBATCH --mem-per-cpu=50G 
#SBATCH --time=0-00:30:00 
#SBATCH --output=my.stdout 
#SBATCH --mail-user=esk81@case.edu 
#SBATCH --mail-type=ALL 
#SBATCH --job-name="rate_survival_experiment" 
module load python 
python adherance_survival_experiment.py