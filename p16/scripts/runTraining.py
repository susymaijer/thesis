#!/usr/bin/python
import os
from datetime import date, datetime
import shared # our own


if __name__ == "__main__":

    # Get the batch id and task id
    task=shared.determine_task_id() 
    print(f"We're going to create a model for task{task}. Is this OK? Enter [y/n].")
    answer=input()
    if answer != "y":
        sys.exit("Abort")
    prev_task=int(task)-1
    print(f"We're going to use the weights from task{prev_task} as the starting weights. Is this OK? Enter [y/n].")
    answer=input()
    pretrained=True
    if answer != "y":
        pretrained=False    
    print(f"We assume this is the first training run for this task and thus are performing preprocessing. Is this OK? Enter [y/n].")
    answer=input()
    preprocess=True
    if answer != "y":
        preprocess=False

    # Get cpu amount
    print(f"How many cpu's do you want to use? Recommended values are [6,8,10].")
    cpu=input()
    
    # Get slurm job file path
    ts=datetime.now().strftime("%Y%m%d%H%M%S")
    job_file=os.path.join(shared.job_dir, f"p16_training_model{task}_{cpu}_{ts}.job")
    
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
        fh.writelines(f"#SBATCH --error={shared.log_dir}/job.%J.training.err\n")
        fh.writelines(f"#SBATCH --output={shared.log_dir}/job.%J.training.out\n\n")
        
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
        fh.writelines(f"source {shared.conda_env_dir}\n")
        fh.writelines("echo \"Verifying environment variables:\"\n")
        fh.writelines("conda env config vars list\n")
        fh.writelines("echo \"Installing hidden layer and nnUnet\"\n\n")
        fh.writelines("python -m pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer\n")
        fh.writelines(f"python -m pip install --editable {shared.nnunet_code_dir}\n")

        # Perform preprocessing
        if preprocess:
            fh.writelines("echo \" Start preprocessing\"")
            fh.writelines(f"nnUNet_plan_and_preprocess -t {task} --verify_dataset_integrity -tl 4 -tf 4")

        if pretrained:
            # Perform training
            fh.writelines("echo \" Start training using prerained weights\"")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 0 -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{pretrained_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_best.model")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 1 -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{pretrained_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/model_best.model")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 2 -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{pretrained_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/model_best.model")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 3 -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{pretrained_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/model_best.model")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 4 -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{pretrained_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/model_best.model")

            # Perform training again so we get filled logfiles
            fh.writelines("echo \"Done training all the folds! Now start the same command but with continue option, to generate log files\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 0 -c --val_disable_overwrite -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{pretrained_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_best.model")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 1 -c --val_disable_overwrite -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{pretrained_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_1/model_best.model")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 2 -c --val_disable_overwrite -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{pretrained_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_2/model_best.model")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 3 -c --val_disable_overwrite -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{pretrained_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_3/model_best.model")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 4 -c --val_disable_overwrite -pretrained_weights {RESULTS_FOLDER}/nnUNet/3d_fullres/Task{pretrained_task}/nnUNetTrainerV2__nnUNetPlansv2.1/fold_4/model_best.model")

        else:
            # Perform training
            fh.writelines("echo \" Start training\"")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 0")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 1")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 2")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 3")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 4")

            # Perform training again so we get filled logfiles
            fh.writelines("echo \"Done training all the folds! Now start the same command but with continue option, to generate log files\n")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 0 -c --val_disable_overwrite")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 1 -c --val_disable_overwrite")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 2 -c --val_disable_overwrite")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 3 -c --val_disable_overwrite")
            fh.writelines(f"nnUNet_train 3d_fullres nnUNetTrainerV2 {task} 4 -c --val_disable_overwrite")

        # Start postprocessing
        fh.writelines("echo \"Start postprocessing\"")
        fh.writelines(f"nnUNet_determine_postprocessing -t {task} -m 3d_fullres -tr nnUNetTrainerV2")  
        fh.writelines("echo \"Program finished with exit code $? at: `\date`\"")
        fh.writelines("")
    
    # Submit batch file
    os.system("sbatch %s" %job_file)
    
