#!/bin/bash


#SBATCH --partition=m40-long
#SBATCH --time=02-00:00:00
#SBATCH --output=output.txt
#SBATCH --error=errors.txt
#SBATCH --gres=gpu:1

export PATH=~/venv_2.7.12/bin:$PATH
module load python/2.7.12
module load cuda80
source ~/venv_2.7.12/bin/activate
python lstm.py 
