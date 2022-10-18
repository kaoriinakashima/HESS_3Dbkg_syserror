#!/bin/bash -l
#
# start 2 MPI processes
#SBATCH --ntasks=2
# requests 8 OpenMP threads per MPI task
#SBATCH --cpus-per-task=8
# do not use hyperthreads
#SBATCH --hint=nomultithread
# allocate nodes for 6 hours
#SBATCH --time=24:00:00
# job name 
#SBATCH --job-name=Testjob
# do not export environment variables
#SBATCH --export=NONE

# do not export environment variables
unset SLURM_EXPORT_ENV




# do not export environment variables
unset SLURM_EXPORT_ENV
conda env list
conda init bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gammapy-0.20
conda info

# set number of threads to requested cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK 
srun --mpi=pmi2 python /home/hpc/caph/mppi045h/3D_analysis/N_parameters_in_L/HESS_3Dbkg_syserror/2-error_in_dataset/5-Tutorial_Nui_Par_Fitting.py

#sbatch.tinyfat submit_tiny_fat.slurm
#squeue.tinyfat