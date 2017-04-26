#!/bin/bash
#
#SBATCH --job-name=it_en_emb
#SBATCH --output=it_en_output.txt  # output file
#SBATCH -e it_en_error.err        # File to which STDERR will be written 
#SBATCH --partition=m40-long
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=8
#SBATCH --nodes=1
#SBATCH --time=1-4:00:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=MaxMemPerCPU    # Memory in MB per cpu allocated
pip install -r requirement.txt --user
python gather_it_en.py
