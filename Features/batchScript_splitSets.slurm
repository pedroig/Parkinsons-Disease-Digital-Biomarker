#!/bin/bash

#SBATCH --job-name=splitSets

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
#SBATCH --time=00:05:00

#SBATCH --mail-user=pao3@rice.edu

#SBATCH --mail-type=ALL

module purge

module load GCC/6.4.0 OpenMPI/2.1.1 Python/3.6.2
pip install --user sklearn

srun python run.py 'splitSets' > output.txt
