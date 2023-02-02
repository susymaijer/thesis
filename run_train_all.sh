#!/bin/bash

echo "Perform training for all folds."
echo ""

read -p "Enter task:" task

read -p "Default partition is LKEBgpu. If not desired, type other partition name:" p
p=${p:-LKEBgpu}

read -p "Default cpu amount is 8. If not desired, type other amount:" cpu
cpu=${cpu:-8}

read -p "Enter config [3d_lowres, 3d_cascade_fullres, 3d_fullres]:" config

read -p "Enter trainer [UNETR,UNETRLarge,Hybrid,Hybrid2,Hybrid2LR,empty]:" trainer

# lowres of fullres
if [ $config != "3d_cascade_fullres" ];
then
   if [ ! -z $trainer ];
   then
      trainer="nnUNetTrainerV2_$trainer"
   fi
   if [ -z $trainer ];
   then
      trainer="nnUNetTrainerV2"
   fi
fi
# cascade
if [ $config == "3d_cascade_fullres" ];
then
   if [ ! -z $trainer ];
   then
      trainer="nnUNetTrainerV2CascadeFullRes_$trainer"
   fi
   if [ -z $trainer ];
   then
      trainer="nnUNetTrainerV2CascadeFullRes"
   fi
fi

job_directory=/home/smaijer/slurm/jobs/
job_file="${job_directory}/train_${task}_${p}_${trainer}_${cpu}_${config}_$(date +"%Y_%m_%d_%I_%M_%p").job"

echo "#!/bin/bash
#SBATCH -J PancreasTrain
#SBATCH -p $p
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpu
#SBATCH --mem=64GB
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --error=/home/smaijer/logs/train/$task/job.%J.err
#SBATCH --output=/home/smaijer/logs/train/$task/job.%J.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=susy.maijer@lumc.nl

# Prepare nnUnet
echo \"Starting at `\date`\"

echo \"Running on hosts: \$SLURM_JOB_NODELIST\"
echo \"Running on \$SLURM_JOB_NUM_NODES nodes.\"
echo \"Running \$SLURM_NTASKS tasks.\"
echo \"CPUs on node: \$SLURM_CPUS_ON_NODE.\"
echo \"Account: \$SLURM_JOB_ACCOUNT\"
echo \"Job ID: \$SLURM_JOB_ID\"
echo \"Job name: \$SLURM_JOB_NAME\"
echo \"Node running script: \$SLURMD_NODENAME\"
echo \"Submit host: \$SLURM_SUBMIT_HOST\"
echo \"GPUS: \$CUDA_VISIBLE_DEVICES or \$SLURM_STEP_GPUS\"
nvidia-smi
echo \"Current working directory is `\pwd`\"

echo "Load all modules.."
module purge
module add system/python/3.10.2
echo "Done with loading all modules. Modules:"
module li
echo "Activate virtualenv pancreasThesis.."
source /exports/lkeb-hpc/smaijer/venv_environments/pancreasThesis/bin/activate
echo "Installing hidden layer and nnUnet.."
python -m pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
python -m pip install -editable /home/smaijer/code/nnUNet

echo "Train all the folds.."
nnUNet_train $config $trainer $task 0
nnUNet_train $config $trainer $task 1
nnUNet_train $config $trainer $task 2
nnUNet_train $config $trainer $task 3
nnUNet_train $config $trainer $task 4

echo "Done training all the folds! Now we do the same command but with continue option, to generate log files".
nnUNet_train $config $trainer $task 0 -c --val_disable_overwrite
nnUNet_train $config $trainer $task 1 -c --val_disable_overwrite
nnUNet_train $config $trainer $task 2 -c --val_disable_overwrite
nnUNet_train $config $trainer $task 3 -c --val_disable_overwrite
nnUNet_train $config $trainer $task 4 -c --val_disable_overwrite

echo \"Program finished with exit code $? at: `\date`\"" > $job_file
sbatch $job_file

