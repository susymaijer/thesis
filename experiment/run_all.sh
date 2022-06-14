#!/bin/bash

echo "Perform everything (preprocess, train, postprocess, inference, eval for all folds."
echo ""

read -p "Enter task:" task

read -p "Default partition is LKEBgpu. If not desired, type other partition name:" p
p=${p:-LKEBgpu}

read -p "Default cpu amount is 8. If not desired, type other amount:" cpu
cpu=${cpu:-8}

read -p "Enter config [3d_lowres, 3d_cascade_fullres, 3d_fullres]:" config


if [ $config != "3d_cascade_fullres" ]; then
   trainer="nnUNetTrainerV2_Transformer"
fi
if [ $config == "3d_cascade_fullres" ]; then
   trainer="nnUNetTrainerV2CascadeFullRes_Transformer"
fi

job_directory=/home/smaijer/slurm/jobs/
job_file="${job_directory}/all_${task}_${p}_${cpu}_${config}_$(date +"%Y_%m_%d_%I_%M_%p").job"

log_path="/home/smaijer/logs/all/$task"
mkdir -p log_path

echo "#!/bin/bash
#SBATCH -J ExpPancreasAll
#SBATCH -p $p
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpu
#SBATCH --mem=32GB
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --error=/home/smaijer/logs/all/$task/job.%J.err
#SBATCH --output=/home/smaijer/logs/all/$task/job.%J.out
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
pip install -e /home/smaijer/experiment/nnUNet

echo "Start preprocessing.."
nnUNet_plan_and_preprocess -t $task --verify_dataset_integrity

echo "Done preprocessing! Start training all the folds.."
nnUNet_train $config $trainer $task 0
nnUNet_train $config $trainer $task 1
nnUNet_train $config $trainer $task 2
nnUNet_train $config $trainer $task 3
nnUNet_train $config $trainer $task 4

echo "Done training all the folds! Now start the same command but with continue option, to generate log files"
nnUNet_train $config $trainer $task 0 -c
nnUNet_train $config $trainer $task 1 -c
nnUNet_train $config $trainer $task 2 -c
nnUNet_train $config $trainer $task 3 -c
nnUNet_train $config $trainer $task 4 -c

echo "Start postprocessing.."
nnUNet_determine_postprocessing -t $task -m $config

echo "Done postprocessing! Now start inferencing its own train and test files."
mkdir -p $OUTPUT/$task/$config/$task/imagesTr
mkdir -p $OUTPUT/$task/$config/$task/imagesTs
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task$task/imagesTr -o $OUTPUT/$task/$config/$task/imagesTr -t $task -m $config
nnUNet_predict -i $nnUNet_raw_data_base/nnUNet_raw_data/Task$task/imagesTs -o $OUTPUT/$task/$config/$task/imagesTs -t $task -m $config

echo "Done inferencing! Now start the evaluation."
nnUNet_evaluate_folder -ref $nnUNet_raw_data_base/nnUNet_raw_data/Task$task/labelsTr -pred $OUTPUT/$task/$config/$task/imagesTr -l 1
nnUNet_evaluate_folder -ref $nnUNet_raw_data_base/nnUNet_raw_data/Task$task/labelsTs -pred $OUTPUT/$task/$config/$task/imagesTs -l 1

echo \"Program finished with exit code $? at: `\date`\"" > $job_file
sbatch $job_file

