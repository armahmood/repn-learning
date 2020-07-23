#!/bin/bash
#SBATCH --array=1-35
#SBATCH --job-name=repn_learning   
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=6
#SBATCH --account=def-ashique
#SBATCH --output=repn_learning%A%a.out
#SBATCH --error=repn_learning%A%a.err

python learner.py --search -f $1 --seeds $SLURM_ARRAY_TASK_ID --save_losses=True