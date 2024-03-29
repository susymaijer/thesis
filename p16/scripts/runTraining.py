#!/usr/bin/python
import os
import sys
from datetime import datetime
import shared 

'''
    This script training of a new model for the most recent P16 task. 
    Default, we use the weights from the previous P16 as the starting weights.
'''

if __name__ == "__main__":

    # Get the task id and the previous task id
    task = shared.determine_task_id() 
    prev_task = int(task)-1

    # Check whether it is correct that we're going to start a new training run
    print(f"We're going to create a model for task{task}. Is this OK? Enter [y/n].")
    answer = input()
    if answer != "y":
        sys.exit("Abort")

    # Check whether we want to use the weights of the previous model as the starting weights
    print(f"We're going to use the weights from task{prev_task} as the starting weights. Is this OK? Enter [y/n].")
    answer = input()
    pretrained = True
    if answer != "y":
        pretrained = False    
    print(f"We assume this is the first training run for this task and thus are performing preprocessing. Is this OK? Enter [y/n].")
    answer = input()
    preprocess = True
    if answer != "y":
        preprocess = False

    # Get cpu amount
    print(f"How many cpu's do you want to use? Recommended values are [6,8,10].")
    cpu = input()
    
    # Get the environment variables
    if len(sys.argv) > 1:
        env_var_dir = sys.argv[1]
    else:
        env_var_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    env_vars = shared.getUserspecificEnvironmentVariables(os.path.join(env_var_dir, "ENVIRONMENT_VARIABLES.txt"))

    # Get slurm job file path
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    job_file = os.path.join(env_vars['job_dir'], f"p16_training_model{task}_{cpu}_{ts}.job")
    RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER')

    # Fill the slurm job file
    with open(job_file, "w+") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -J P16panc_training\n")
        fh.writelines("#SBATCH -p LKEBgpu\n")
        fh.writelines("#SBATCH -N 1\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines(f"#SBATCH --cpus-per-task={cpu}\n")
        fh.writelines("#SBATCH --mem=64GB\n")
        fh.writelines("#SBATCH --gres=gpu:RTX6000:1\n")
        fh.writelines(f"#SBATCH --error={env_vars['log_dir']}/job.%J.training.err\n")
        fh.writelines(f"#SBATCH --output={env_vars['log_dir']}/job.%J.training.out\n\n")
        
        fh.writelines("echo \"Starting at `\date`\"\n")
        fh.writelines("echo \"Running on hosts: \$SLURM_JOB_NODELIST\"\n")
        fh.writelines("echo \"Running on \$SLURM_JOB_NUM_NODES nodes.\"\n")
        fh.writelines("echo \"Running \$SLURM_NTASKS tasks.\"\n")
        fh.writelines("echo \"CPUs on node: \$SLURM_CPUS_ON_NODE.\"\n")
        fh.writelines("echo \"Account: \$SLURM_JOB_ACCOUNT\"\n")
        fh.writelines("echo \"Job ID: \$SLURM_JOB_ID\"\n")
        fh.writelines("echo \"Job name: \$SLURM_JOB_NAME\"\n")
        fh.writelines("echo \"Node running script: \$SLURMD_NODENAME\"\n")
        fh.writelines("echo \"Submit host: \$SLURM_SUBMIT_HOST\"\n")
        fh.writelines("echo \"GPUS: \$CUDA_VISIBLE_DEVICES or \$SLURM_STEP_GPUS\"\n")
        fh.writelines("nvidia-smi\n")
        fh.writelines("echo \"Current working directory is `\pwd`\"\n\n")

        fh.writelines("echo \"Load all modules..\"\n")
        fh.writelines("module purge\n")
        fh.writelines("module add system/python/3.10.2\n")
        fh.writelines("echo \"Done with loading all modules. Modules:\"\n")
        fh.writelines("module li\n")
        fh.writelines("echo \"Activate conda env nnunet\"\n")
        fh.writelines(f"source {env_vars['conda_env_dir']}\n")
        fh.writelines("echo \"Verifying environment variables:\"\n")
        fh.writelines("conda env config vars list\n")
        fh.writelines("echo \"Installing hidden layer and nnUnet\"\n\n")
        fh.writelines("python -m pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer\n")
        fh.writelines(f"python -m pip install --editable {env_vars['nnUNet_code_dir']}\n")

        # Perform preprocessing
        if preprocess:
            fh.writelines("echo \" Start preprocessing\"\n\n")
            fh.writelines(f"nnUNet_plan_and_preprocess -t {task} --verify_dataset_integrity -tl 4 -tf 4\n")

        if pretrained:
            # Perform training with pretrained starting weights
            fh.writelines("echo \" Start training using prerained weights\"\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 0 -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{prev_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_best.model\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 1 -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{prev_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/model_best.model\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 2 -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{prev_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/model_best.model\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 3 -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{prev_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/model_best.model\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 4 -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{prev_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/model_best.model\n")

            # Perform training again so we get filled debug.json files within $RESULTS folder
            fh.writelines("echo \"Done training all the folds! Now start the same command but with continue option, to generate log files\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 0 -c --val_disable_overwrite -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{prev_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_best.model\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 1 -c --val_disable_overwrite -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{prev_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/model_best.model\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 2 -c --val_disable_overwrite -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{prev_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/model_best.model\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 3 -c --val_disable_overwrite -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{prev_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/model_best.model\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 4 -c --val_disable_overwrite -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{prev_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/model_best.model\n")

        else:
            # Perform training
            fh.writelines("echo \" Start training\"\n\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 0\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 1\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 2\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 3\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 4\n")

            # Perform training again so we get filled logfiles
            fh.writelines("echo \"Done training all the folds! Now start the same command but with continue option, to generate log files\"\n\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 0 -c --val_disable_overwrite\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 1 -c --val_disable_overwrite\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 2 -c --val_disable_overwrite\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 3 -c --val_disable_overwrite\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 4 -c --val_disable_overwrite\n")

        # Start postprocessing
        fh.writelines("echo \"Start postprocessing\"\n\n")
        fh.writelines(f"nnUNet_determine_postprocessing -t {task} -m 3d_fullres -tr nnUNetTrainerV2\n")  
        fh.writelines("echo \"Program finished with exit code $? at: `\date`\"\\n\n")
        fh.writelines("")
    
    # Submit batch file
    os.system("sbatch %s" %job_file)
    
