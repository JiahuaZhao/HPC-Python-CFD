#!/bin/bash --login

#SBATCH --job-name=P-NUMPY
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=pr1uzhao
#SBATCH --partition=standard
#SBATCH --qos=standard

# Setup the batch environment
module load epcc-job-env
export PYTHONUSERBASE=/work/pr1ushpc/pr1ushpc/pr1uzhao/.local
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH

# Out of memory
module load cray-python
module swap cray-mpich cray-mpich/8.1.3
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH

# srun launches the parallel program based on the SBATCH options
python3.8 cfd_numpy.py 256 5000 2
