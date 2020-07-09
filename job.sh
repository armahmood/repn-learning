#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-ashique
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

bash run.sh -r 30 -f "100 1000" -s "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29" -x 1