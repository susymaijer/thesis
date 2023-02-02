#!/bin/bash

echo "Perform preprocessing (with GPU) for a certain task. Only performed once!"
echo ""

read -p "Enter task:" task

read -p "Default cpu amount is 8. If not desired, type other amount:" cpu
cpu=${cpu:-8}

read -p "Default wall time is 01:00:00. If not desired, type other wall time:" t
t=${t:-01:00:00}

read -p "Enter plans (if any):" plan

read -p "Enter plan identifier (if any):" ident

# We assume running this from the script directory
job_directory=/home/smaijer/slurm/jobs/
job_file="${job_directory}/preprocess_${task}_${cpu}_$(date +"%Y_%m_%d_%I_%M_%p").job"

echo "#!/bin/bash
#SBATCH -J PancreasPreprocess
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpu
#SBATCH --time=$t
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --error=/home/smaijer/logs/preprocess/$task/job.%J.err
#SBATCH --output=/home/smaijer/logs/preprocess/$task/job.%J.out
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

if [ -z $plan ]
then
    nnUNet_plan_and_preprocess -t $task --verify_dataset_integrity -tl 6 -tf 6
else
    echo "Use specific plans"
    nnUNet_plan_and_preprocess -t $task --verify_dataset_integrity -tl 6 -tf 6 -overwrite_plans $plan -overwrite_plans_identifier $ident -pl3d ExperimentPlanner3D_v21_Pretrained
fi

echo \"Program finished with exit code $? at: `\date`\"" > $job_file
sbatch $job_file

