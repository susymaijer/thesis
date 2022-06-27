#!/bin/bash
#SBATCH -J init
##SBATCH -p LKEBgpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --error=/home/smaijer/logs/init/job.%J.err
#SBATCH --output=/home/smaijer/logs/init/job.%J.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=susy.maijer@lumc.nl

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
conda activate nn2
echo "Verifying environment variables:"
conda env config vars list
echo "Installing pytorch.."| 
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
echo "Verify pytorch"
python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'
echo "Installing nnU-net.."
pip install -e /home/smaijer/code/nnUNet

