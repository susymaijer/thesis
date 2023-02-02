#!/bin/bash

echo "Make predictions for MSD scans of different sizes (3x small, 3x medium, 3x large). Use the default MSD model."
echo "We added printing statements to the steps, so we can analyse how long each step takes."
echo ""

read -p "Default cpu amount is 6. If not desired, type other amount:" cpu
cpu=${cpu:-6}

read -p "Default wall time is 01:00:00. If not desired, type other wall time:" t
t=${t:-01:00:00}

read -p "Enter folds, like '0 1 2 3 4':" folds
t=${t:-0 1 2 3 4}

read -p "Enter name:" name

# We assume running this from the script directory
job_directory=/home/smaijer/slurm/jobs/
job_file="${job_directory}/test_inference_${name}_$(date +"%Y_%m_%d_%I_%M_%p").job"

echo "#!/bin/bash
#SBATCH -J test
#SBATCH -p LKEBgpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpu
#SBATCH --time=$t
#SBATCH --mem=85GB
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --error=$TEST/logs/$name.%J.err
#SBATCH --output=$TEST/logs/$name.%J.out
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

mkdir -p $TEST/$name/small
mkdir -p $TEST/$name/medium
mkdir -p $TEST/$name/big

echo "We start with small images"
nnUNet_predict -i $TEST/cases/images/small -o $TEST/$name/small -t 510 -m 3d_fullres -tr nnUNetTrainerV2 -f $folds
nnUNet_evaluate_folder -ref $TEST/cases/labels/small -pred $TEST/$name/small -l 1

echo "Now onto medium images"
nnUNet_predict -i $TEST/cases/images/medium -o $TEST/$name/medium -t 510 -m 3d_fullres -tr nnUNetTrainerV2 -f $folds
nnUNet_evaluate_folder -ref $TEST/cases/labels/medium -pred $TEST/$name/medium -l 1

echo "And finally big images"
nnUNet_predict -i $TEST/cases/images/big -o $TEST/$name/big -t 510 -m 3d_fullres -tr nnUNetTrainerV2r -f $folds
nnUNet_evaluate_folder -ref $TEST/cases/labels/big -pred $TEST/$name/big -l 1

sstat $SLURM_JOB_ID
sacct -l -j $SLURM_JOB_ID

echo \"Program finished with exit code $? at: `\date`\"" > $job_file
sbatch $job_file

