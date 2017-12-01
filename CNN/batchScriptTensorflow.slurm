#!/bin/bash

#SBATCH --job-name=CNNmodel

# This value can be adjusted to run in a condo if applicable
# or commons if the max walltime is more than 4 hours
#SBATCH --partition=commons

#SBATCH --account=commons

#SBATCH --ntasks=1

# Tell SLURM to only place threads on physical cores, leave this value as is
#SBATCH --threads-per-core=1

# This value can be adjusted if a different thread count is required
#SBATCH --cpus-per-task=1

# This value can be adjusted if a different amount of memory is required
#SBATCH --mem-per-cpu=8G

# This value can be adjusted if a different maximum wall time is required
#SBATCH --time=24:00:00

#SBATCH --mail-user=pao3@rice.edu

#SBATCH --mail-type=ALL

module purge

# load the GCC-OpenMPI-MKL-CUDA toolchain and the default TensorFlow module
module load gomkl/2017a TensorFlow

pip install sklearn --user
pip install pywavelets --user

srun python cnnModel.py > output.txt