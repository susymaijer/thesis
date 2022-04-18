#!/bin/bash

#SBATCH -J NIHPancreasTrain
##SBATCH -p LKEBgpu
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --error=/home/smaijer/logs/other/job.%J.err
#SBATCH --output=/home/smaijer/logs/other/job.%J.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=susy.maijer@lumc.nl

# Prepare nnUnet
echo "Starting at `date`"

echo "Running on hosts: $SLURM_JOB_NODELIST"
echo "Running on $SLURM_JOB_NUM_NODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node running script: $SLURMD_NODENAME"
echo "Submit host: $SLURM_SUBMIT_HOST"
echo "GPUS: $CUDA_VISIBLE_DEVICES or $SLURM_STEP_GPUS"
nvidia-smi
echo "Current working directory is `pwd`"

echo "Load all modules.."
module purge
module add tools/miniconda/python3.8/4.9.2
echo "Done with loading all modules. Modules:"
module li
echo "Activate conda env nnunet.."
conda activate nn
echo "Verifying environment variables:"
conda env config vars list
echo "Installing nnU-net.."
pip install -e /home/smaijer/code/nnUNet

python -c 'import torch;print(torch.backends.cudnn.version());print(torch.__version__);'
nnUNet_train 2d nnUNetTrainerV2_5epochs 500 0
nnUNet_train 3d_fullres nnUNetTrainerV2_5epochs 500 0
nnUNet_train 3d_fullres nnUNetTrainerV2_5epochs_dummyLoad 500 0

echo "Program finished with exit code $? at: `date`"

