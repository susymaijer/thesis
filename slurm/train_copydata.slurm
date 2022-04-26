#!/bin/bash
#SBATCH -J NIHPancreasTrain
#SBATCH -p LKEBgpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00 
#SBATCH --mem=32GB
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --error=/home/smaijer/logs/train/nih/job.%J.err
#SBATCH --output=/home/smaijer/logs/train/nih/job.%J.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=susy.maijer@lumc.nl

# Prepare nnUnet
echo "Starting at `date`"

echo "Running on hosts: $SLURM_JOB_NODELIST"
echo "Running on $SLURM_JOB_NUM_NODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "CPUs on node: $SLURM_CPUS_ON_NODE."
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node running script: $SLURMD_NODENAME"
echo "Submit host: $SLURM_SUBMIT_HOST"
echo "GPUS: $CUDA_VISIBLE_DEVICES or $SLURM_STEP_GPUS"
nvidia-smi
echo "Current working directory is `pwd`"
dir=/tmp/$SLURM_JOB_NAME
echo "Creating $dir"
mkdir $dir
mkdir $dir/Task500_NIH_Pancreas
mkdir $dir/Task500_NIH_Pancreas/nnUNetData_plans_v2.1_stage0

echo "Load all modules.."
module purge
module add tools/miniconda/python3.8/4.9.2
echo "Done with loading all modules. Modules:"
module li
echo "Activate conda env nnunet.."
conda activate nn
echo "Verifying environment variables:"
conda env config vars list
old_dir=$nnUNet_preprocessed
echo "Overwrite preprocessed variable.."
conda env config vars set nnUNet_preprocessed=$dir
conda activate nn
conda env config vars list
echo "Move preprocessed data to new temporary dir.."
cp $old_dir/Task500_NIH_Pancreas/nnUNetData_plans_v2.1_stage0 $dir/Task500_NIH_Pancreas/ -r
cp $old_dir/Task500_NIH_Pancreas/nnUNetPlansv2.1_plans_3D.pkl $dir/Task500_NIH_Pancreas/ -r
cp $old_dir/Task500_NIH_Pancreas/splits_final.pkl $dir/Task500_NIH_Pancreas/ -r

echo "Installing nnU-net.."
pip install -e /home/smaijer/code/nnUNet

nnUNet_train 3d_lowres nnUNetTrainerV2 500 4 -c --npz

echo "Remove data"
rm $dir -r

echo "Program finished with exit code $? at: `date`"

