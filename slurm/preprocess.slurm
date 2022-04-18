#!/bin/bash
#SBATCH -J NIHPancreasPreprocess
##SBATCH -p LKEBgpu
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --error=/home/smaijer/logs/preprocess/nih/job.%J.err
#SBATCH --output=/home/smaijer/logs/preprocess/nih/job.%J.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=susy.maijer@lumc.nl

# Prepare nnUnet
#/home/smaijer/init.sh

#----------------TODO shell scriptje aanroepen
#./preprocess.sh 500

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

nnUNet_plan_and_preprocess -t 500 --verify_dataset_integrity -tl 8 -tf 8

echo "Program finished with exit code $? at: `date`"

