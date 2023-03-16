#!/bin/bash

#- Job parameters

# (TODO)
# Please modify job name

#SBATCH -J nrs              # The job name
#SBATCH -o ./%j.out        # Write the standard output to file named 'ret-<job_number>.out'
#SBATCH -e ./%j.err        # Write the standard error to file named 'ret-<job_number>.err'


#- Resources

# (TODO)
# Please modify your requirements

#SBATCH -p nv-gpu-hw              # Submit to 'nv-gpu' Partitiion
#SBATCH -t 1-06:00:00                # Run for a maximum time of 0 days, 12 hours, 00 mins, 00 secs
#SBATCH --nodes=1                  # Request N nodes
#SBATCH --gres=gpu:8            # Request M GPU per node
#SBATCH --gres-flags=enforce-binding # CPU-GPU Affinity
#SBATCH --qos=gpu-normal           # Request QOS Type

###
### The system will alloc 8 or 16 cores per gpu by default.
### If you need more or less, use following:
### #SBATCH --cpus-per-task=K            # Request K cores
###
### 
### Without specifying the constraint, any available nodes that meet the requirement will be allocated
### You can specify the characteristics of the compute nodes, and even the names of the compute nodes
###
### #SBATCH --constraint="A100|RTX8000" # Request GPU Type: Volta(V100 or V100S) or RTX8000
###

#- Log information

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Job run at:"
echo "$(hostnamectl)"

#- Load environments
source /tools/module_env.sh
module list                       # list modules loaded

##- Tools
module load cluster-tools/v1.0
module load slurm-tools/v1.0
module load cmake/3.15.7
module load git/2.17.1
module load vim/8.1.2424

##- language
module load python3/3.6.8

##- CUDA
module load cuda-cudnn/11.1-8.1.1

##- virtualenv
# source xxxxx/activate

echo $(module list)              # list modules loaded
echo $(which gcc)
echo $(which python)
echo $(which python3)

cluster-quota                    # nas quota

nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit # gpu info

#- Warning! Please not change your CUDA_VISIBLE_DEVICES
#- in `.bashrc`, `env.sh`, or your job script
echo "Use GPU ${CUDA_VISIBLE_DEVICES}"                              # which gpus
#- The CUDA_VISIBLE_DEVICES variable is assigned and specified by SLURM

#- Job step
#divide
#/workspace/S/liyutai/anaconda3/envs/nrs/bin/python divide.py

#reduce
#/workspace/S/liyutai/anaconda3/envs/nrs/bin/python reduce.py

#RNN
#/workspace/S/liyutai/anaconda3/envs/nrs/bin/python expression_handle.py
#/workspace/S/liyutai/anaconda3/envs/nrs/bin/python expression_handle_combine.py
#/workspace/S/liyutai/anaconda3/envs/nrs/bin/python expression_feature.py
/lustre/S/gaomj/miniconda3/envs/bachelor/bin/python trainer.py
#/workspace/S/liyutai/anaconda3/envs/nrs/bin/python main.py

#- End
echo "Job end at $(date "+%Y-%m-%d %H:%M:%S")"
