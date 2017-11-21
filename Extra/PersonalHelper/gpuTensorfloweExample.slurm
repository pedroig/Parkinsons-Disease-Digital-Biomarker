#!/bin/bash
# This value can be adjusted to run in a condo if applicable
# or commons if the max walltime is more than 4 hours
#SBATCH --partition=scavenge
# Leave this value as is unless running in a condo
#SBATCH --account=commons
# To run SMP applications only launch one task, leave this value as is
#SBATCH --ntasks=1
# Tell SLURM to only place threads on physical cores, leave this value as is
#SBATCH --threads-per-core=1
# This value can be adjusted if a different thread count is required
#SBATCH --cpus-per-task=4
# This value can be adjusted if a different GPU count is required
#SBATCH --gres=gpu:1
# This value can be adjusted if a different amount of memory is required
#SBATCH --mem-per-cpu=1G
# This value can be adjusted if a different maximum wall time is required
#SBATCH --time=00:30:00

# purge any modules that might be automatically loaded
module purge
# load the GCC-OpenMPI-MKL-CUDA toolchain and the default TensorFlow module
module load gomklc/2017a TensorFlow

# launch python with your input file
srun python helloTF.py