#!/bin/bash
#SBATCH --job-name fed_train
#SBATCH --ntasks 1
#SBATCH --time 20:00:00
#SBATCH --partition allgroups
#SBATCH --output output/out_train.txt
#SBATCH --error output/err_train.txt
#SBATCH --mem=12G
#SBATCH --gres=gpu:a40:1

# Comandi da eseguire

# Attiva enviroment
source ~/.bashrc
conda activate foundation-fed

# Naviga nella directory
cd /nfsd/lttm4/tesisti/Bortolatto/codice/progetto1

# Esegui lo script
python3 train.py
