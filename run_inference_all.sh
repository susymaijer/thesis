#!/bin/bash

echo "Make predictions for a specific folder. We do this either for training images or test images!"
echo ""

read -p "Enter task of model:" task

read -p "Enter task of inference imagesTr/imagesTs folder:" taskPredict

read -p "Default partition is LKEBgpu. If not desired, type other partition name:" p
p=${p:-LKEBgpu}

read -p "Default cpu amount is 6. If not desired, type other amount:" cpu
cpu=${cpu:-6}

read -p "Default wall time is 01:00:00. If not desired, type other wall time:" t
t=${t:-01:00:00}

read -p "Enter config [3d_lowres, 3d_cascade_fullres, 3d_fullres]:" config

read -p "Enter trainer [UNETR,UNETRLarge,Hybrid,empty]:" trainer

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

# We assume running this from the script directory
job_directory=/home/smaijer/slurm/jobs/
job_file="${job_directory}/inference_eval_${task}_${taskPredict}_${p}_${trainer}_${cpu}_${config}_$(date +"%Y_%m_%d_%I_%M_%p").job"

echo "#!/bin/bash
#SBATCH -J PancreasInferenceEval
#SBATCH -p $p
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpu
#SBATCH --time=$t
#SBATCH --mem=32GB
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --error=/home/smaijer/logs/inference/$task/job.%J.err
#SBATCH --output=/home/smaijer/logs/inference/$task/job.%J.out
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

echo \"Load all modules..\"
module purge
module add tools/miniconda/python3.8/4.9.2
echo \"Done with loading all modules. Modules:\"
module li
echo \"Activate conda env nnunet..\"
conda activate nn
echo \"Verifying environment variables:\"
conda env config vars list
echo \"Installing nnU-net..\"
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
pip install --upgrade -e /home/smaijer/nnUNet

mkdir -p $OUTPUT/$task/$config/$trainer/$taskPredict/imagesTr
mkdir -p $OUTPUT/$task/$config/$trainer/$taskPredict/imagesTs

nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task$taskPredict/imagesTr -o $OUTPUT/$task/$config/$trainer/$taskPredict/imagesTr -t $task -m $config -tr $trainer -ctr "TODO"
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task$taskPredict/imagesTs -o $OUTPUT/$task/$config/$trainer/$taskPredict/imagesTs -t $task -m $config -tr $trainer -ctr "TODO"

nnUNet_evaluate_folder -ref $nnUNet_raw_data_base/nnUNet_raw_data/Task$taskPredict/labelsTr -pred $OUTPUT/$task/$config/$trainer/$taskPredict/imagesTr -l 1
nnUNet_evaluate_folder -ref $nnUNet_raw_data_base/nnUNet_raw_data/Task$taskPredict/labelsTs -pred $OUTPUT/$task/$config/$trainer/$taskPredict/imagesTs -l 1


echo \"Program finished with exit code $? at: `\date`\"" > $job_file
sbatch $job_file

