#!/bin/bash

echo "Evaluate a specific folder (either test cases or training cases)."

read -p "Enter task (model):" task

read -p "Enter task (evaluation folder)": taskPredict

read -p "Default cpu amount is 6. If not desired, type other amount:" cpu
cpu=${cpu:-6}

read -p "Default wall time is 00:30:00. If not desired, type other wall time:" t
t=${t:-00:30:00}

read -p "Enter config [3d_lowres, 3d_cascade_fullres, 3d_fullres], default 3d_fullres:" config
config=${config:-3d_fullres}
read -p "Enter trainer [UNETR,UNETRLarge,Hybrid,empty]:" trainer

read -p "Enter folder suffix [Ts, Tr]:" tstr

read -p "Enter labels (default 1 2 3 4 5 6 7 8 9 10 11 12 13):" lab
lab=${lab:-1 2 3 4 5 6 7 8 9 10 11 12 13}

read -p "Enter 'y' if it's relabeled:" relabeled
relabeled=${relabeled:-n}

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

if [ $relabeled == "y" ]; 
then 
  path=$OUTPUT/$task/$config/$trainer/$taskPredict/images$tstr/relabeled 
fi
if [ $relabeled != "y" ];
then 
 path=$OUTPUT/$task/$config/$trainer/$taskPredict/images$tstr
fi 

# We assume running this from the script directory
job_directory=/home/smaijer/slurm/jobs/
job_file="${job_directory}/evalfolder_${task}_${config}_${trainer}_${taskPredict}_${tstr}_${cpu}_$(date +"%Y_%m_%d_%I_%M_%p").job"

echo "#!/bin/bash
#SBATCH -J PancreasEvalFolder
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpu
#SBATCH --time=$t
#SBATCH --mem=12GB
#SBATCH --error=/home/smaijer/logs/evalfolder/$task/job.%J.err
#SBATCH --output=/home/smaijer/logs/evalfolder/$task/job.%J.out
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
echo "Activate conda env nnunet.."
source /exports/lkeb-hpc/smaijer/venv_environments/pancreasThesis/bin/activate
echo "Verifying environment variables:"
conda env config vars list
echo "Installing hidden layer and nnUnet.."
python -m pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
python -m pip install -editable /home/smaijer/code/nnUNet

nnUNet_evaluate_folder -ref $nnUNet_raw_data_base/nnUNet_raw_data/Task$taskPredict/labels$tstr -pred $path -l $lab

echo \"Program finished with exit code $? at: `\date`\"" > $job_file
sbatch $job_file

