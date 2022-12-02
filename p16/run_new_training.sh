#!/bin/bash

echo "Perform everything (preprocess, train, postprocess, inference) for all folds."
echo ""

read -p "Enter task:" task

read -p "Perform preprocessing? Press 'y' if this is the first time training this task, otherwise press enter:" prep

read -p "Default partition is LKEBgpu. If not desired, type other partition name:" p
p=${p:-LKEBgpu}

read -p "Default cpu amount is 8. If not desired, type other amount:" cpu
cpu=${cpu:-8}

read -p "Default config is 3d_fullres. If not desired, enter other config [3d_lowres, 3d_cascade_fullres, 3d_fullres]:" config
config=${config:-3d_fullres}

read -p "If desired, enter path to pretrained weights:" pretrained

read -p "If desired, enter plans identifier:" identifier

# lowres of fullres
if [ "$config" != "3d_cascade_fullres" ];
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
if [ "$config" = "3d_cascade_fullres" ];
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

source ENVIRONMENT_VARIABLES.txt
echo "job_dir is $job_dir"
echo "log_dir is $log_dir"
echo "conda_env_dir is $conda_env_dir"
echo "nnunet_code_dir is $nnUNet_code_dir"
echo ""

job_file="${job_dir}/trainp16_${task}_${p}_${trainer}_${cpu}_${config}_$(date +"%Y_%m_%d_%I_%M_%p").job"
log_path="${log_dir}/$task"
mkdir -p log_path

echo "#!/bin/bash
#SBATCH -J PancreasTrain
#SBATCH -p $p
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$cpu
#SBATCH --mem=64GB
#SBATCH --gres=gpu:RTX6000:1
#SBATCH --error=$log_dir/job_train_$task.%J.err
#SBATCH --output=$log_dir/job_train_$task.%J.out
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


echo "Start preprocessing.."
if [ $prep == "y" ];
then
   nnUNet_plan_and_preprocess -t $task --verify_dataset_integrity -tl 4 -tf 4
fi

echo "Done preprocessing! Start training all the folds.."
if [ -z $pretrained ];
then

    if [ ! -z $identifier ];
    then
         echo "Run with specific identifier $identifier"
         nnUNet_train $config $trainer $task 0 -p $identifier
         nnUNet_train $config $trainer $task 1 -p $identifier
         nnUNet_train $config $trainer $task 2 -p $identifier
         nnUNet_train $config $trainer $task 3 -p $identifier
         nnUNet_train $config $trainer $task 4 -p $identifier
    else
        nnUNet_train $config $trainer $task 0
         nnUNet_train $config $trainer $task 1
         nnUNet_train $config $trainer $task 2
         nnUNet_train $config $trainer $task 3
         nnUNet_train $config $trainer $task 4 
    fi

    echo "Done training all the folds! Now start the same command but with continue option, to generate log files"
    
    if [ ! -z $identifier ];
    then
	 nnUNet_train $config $trainer $task 0 -p $identifier -c --val_disable_overwrite
         nnUNet_train $config $trainer $task 1 -p $identifier -c --val_disable_overwrite
         nnUNet_train $config $trainer $task 2 -p $identifier -c --val_disable_overwrite
         nnUNet_train $config $trainer $task 3 -p $identifier -c --val_disable_overwrite
         nnUNet_train $config $trainer $task 4 -p $identifier -c --val_disable_overwrite
    else
         nnUNet_train $config $trainer $task 0 -c --val_disable_overwrite
         nnUNet_train $config $trainer $task 1 -c --val_disable_overwrite
         nnUNet_train $config $trainer $task 2 -c --val_disable_overwrite
         nnUNet_train $config $trainer $task 3 -c --val_disable_overwrite
         nnUNet_train $config $trainer $task 4 -c --val_disable_overwrite
    fi

else
    echo "Use pretrained!"
    nnUNet_train $config $trainer $task 0 -pretrained_weights $pretrained/fold_0/model_best.model
    nnUNet_train $config $trainer $task 1 -pretrained_weights $pretrained/fold_1/model_best.model
    nnUNet_train $config $trainer $task 2 -pretrained_weights $pretrained/fold_2/model_best.model
    nnUNet_train $config $trainer $task 3 -pretrained_weights $pretrained/fold_3/model_best.model
    nnUNet_train $config $trainer $task 4 -pretrained_weights $pretrained/fold_4/model_best.model

    echo "Done training all the folds! Now start the same command but with continue option, to generate log files"
    nnUNet_train $config $trainer $task 0 -c -pretrained_weights $pretrained/fold_0/model_best.model --val_disable_overwrite
    nnUNet_train $config $trainer $task 1 -c -pretrained_weights $pretrained/fold_1/model_best.model --val_disable_overwrite
    nnUNet_train $config $trainer $task 2 -c -pretrained_weights $pretrained/fold_2/model_best.model --val_disable_overwrite
    nnUNet_train $config $trainer $task 3 -c -pretrained_weights $pretrained/fold_3/model_best.model --val_disable_overwrite
    nnUNet_train $config $trainer $task 4 -c -pretrained_weights $pretrained/fold_4/model_best.model --val_disable_overwrite
fi

echo "Start postprocessing.."
if [ -z $identifier ];
then
    echo "Postprocessing with default plans"
    nnUNet_determine_postprocessing -t $task -m $config -tr $trainer
else
    echo "Postprocessing with pretrained"
    nnUNet_determine_postprocessing -t $task -m $config -tr $trainer -p $identifier
fi

echo \"Program finished with exit code $? at: `\date`\"" > $job_file
sbatch $job_file

