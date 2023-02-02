#!/bin/bash

echo "Make predictions for test cases of folders."
echo "We do this as part of a performance analysis, where we analyse the effects of changing:"
echo "- the mode"
echo "- the amount of folds [1 or all]"
echo "- disabling TTA"
echo ""

read -p "Enter task of model:" task

read -p "Enter task of inference imagesTr/imagesTs folder:" taskPredict

read -p "Default cpu amount is 8. If not desired, type other amount:" cpu
cpu=${cpu:-8}

read -p "Default wall time is 01:00:00. If not desired, type other wall time:" t
t=${t:-01:00:00}

read -p "Enter folds, default is '0 1 2 3 4':" folds
folds=${folds:-0 1 2 3 4}

read -p "Enter foldsname for printing, default is '01234':" foldname
foldname=${foldname:-01234}

read -p "Enter mode, default is 'normal' (otherwise choose 'fast', 'fastest'):" mode
mode=${mode:-normal}

read -p "Disable tta? [y,n]" disable_tta

if [ $disable_tta == "y" ];
then
	tta="--disable_tta"
fi

# create dir
output_dir="$OUTPUT/$task/$taskPredict/${cpu}_${mode}_${foldname}_${disable_tta}/imagesTs"
mkdir -p $output_dir

# We assume running this from the script directory
job_directory=/home/smaijer/slurm/jobs/
job_file="${job_directory}/inference_eval_${task}_${taskPredict}_${cpu}_${mode}_${foldname}_${disable_tta}_$(date +"%Y_%m_%d_%I_%M_%p").job"

echo "#!/bin/bash
#SBATCH -J PancreasExp
#SBATCH -p LKEBgpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpu
#SBATCH --time=$t
#SBATCH --mem=85GB
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
python -m pip install --editable /home/smaijer/code/nnUNet

nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task$taskPredict/imagesTs -o $output_dir -t $task -m 3d_fullres -tr nnUNetTrainerV2 -f $folds --mode $mode $tta

nnUNet_evaluate_folder -ref $nnUNet_raw_data_base/nnUNet_raw_data/Task$taskPredict/labelsTs -pred $output_dir -l 1

echo \"Program finished with exit code $? at: `\date`\"" > $job_file
sbatch $job_file

