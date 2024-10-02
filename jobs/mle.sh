#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --account=mp107e
#SBATCH --nodes=8
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=12
#SBATCH -J SOLAT
#SBATCH -o solat.out
#SBATCH -e solat.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alonappan@ucsd.edu

module load python
conda activate cb
cd /global/homes/l/lonappan/workspace/solat_cb/jobs

mpirun -np $SLURM_NTASKS python run.py