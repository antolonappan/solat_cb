#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --nodes=4
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH -J SOLAT
#SBATCH -o solat.out
#SBATCH -e solat.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=alonappan@ucsd.edu


#source /global/homes/l/lonappan/.bashrc
module load python
conda activate cb
cd /global/homes/l/lonappan/workspace/solat_cb/jobs

mpirun -np $SLURM_NTASKS python run.py