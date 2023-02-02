#!/bin/bash

echo "Make predictions for CT scans for Alexander Broersen (not part of thesis)."
echo ""

read -p "Enter task of model:" task

read -p "Enter foldername:" folder

read -p "Default partition is LKEBgpu. If not desired, type other partition name:" p
p=${p:-LKEBgpu}

read -p "Default cpu amount is 6. If not desired, type other amount:" cpu
cpu=${cpu:-6}

read -p "Default wall time is 01:00:00. If not desired, type other wall time:" t
t=${t:-01:00:00}

read -p "Disable tta? Type 'y' or 'n':" disable_tta

# Disable tta
if [ $disable_tta == "y" ];
then
        tta="--disable_tta"
fi

# We assume running this from the script directory
job_directory=/home/smaijer/slurm/jobs/
job_file="${job_directory}/inference_eval_${task}_${folder}_${p}_${cpu}_${foldname}_${disable_tta}_$(date +"%Y_%m_%d_%I_%M_%p").job"

echo "#!/bin/bash
#SBATCH -J Inference
#SBATCH -p $p
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
echo "Activate virtualenv pancreasThesis.."
source /exports/lkeb-hpc/smaijer/venv_environments/pancreasThesis/bin/activate
echo "Installing hidden layer and nnUnet.."
python -m pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer
python -m pip install --editable /home/smaijer/code/nnUNet

mkdir -p $nnUNet_raw_data_base/alex/$folder/segmentations/$task/$foldname/$disable_tta

nnUNet_predict -i $nnUNet_raw_data_base/alex/$folder/scans -o $nnUNet_raw_data_base/alex/$folder/segmentations/$task/$foldname/$disable_tta -t $task -m 3d_fullres -tr nnUNetTrainerV2 -f 0 1 2 3 4 $tta

echo \"Program finished with exit code $? at: `\date`\"" > $job_file
sbatch $job_file

