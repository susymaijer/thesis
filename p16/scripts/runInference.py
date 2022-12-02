#!/usr/bin/python
import os
import sys
from datetime import date, datetime
# custom module
import shared

'''
   PREREQUISITE:
   - The p16/batchX/niftis folder should be present. Here, batchX is the newest batch.
     This folder should be automatically created and filled if the script createP16Dataset.py has been executed.


   GOAL OF SCRIPT:
   - This script creates automated segmentations of the scans of the newest batch.

'''

if __name__ == "__main__":

    # Get the batch id and task id
    batch=shared.determine_batch_id()
    task=shared.determine_task_id() - 1 # the most current task contains the niftis of our batch and does not have an associated model yet
    print(f"We're going to make predictions on batch {batch} with model {task}. Is this OK? Enter [y/n].")
    answer=input()
    if answer != "y":
        print("Which model would you like to use instead?")
        task=input()
        print(f"We're going to make predictions with the model of task {task} instead. Is this OK? Enter [y/n]")
        answer=input()
        if answer != "y":
            sys.exit("Abort")

    # Get cpu amount
    print(f"How many cpu's do you want to use? Recommended values are 4 / 6 / 8.")
    cpu=input()

    # Get the environment variables
    if len(sys.argv) > 1:
        env_var_dir=sys.argv[1]
    else:
        env_var_dir=os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    env_vars=shared.getUserspecificEnvironmentVariables(os.path.join(env_var_dir, "ENVIRONMENT_VARIABLES.txt"))

    # Define the paths to the folder containing the niftis which we are going to predict
    batch_dir=os.path.join(env_vars['p16_dir'], f'batch{batch}')
    niftis_dir=os.path.join(batch_dir, 'niftis')

    # Get slurm job file path
    ts=datetime.now().strftime("%Y%m%d%H%M%S")
    job_file=os.path.join(env_vars['job_dir'], f"p16_inference_model{task}_batch{batch}_{ts}.job")
    
    # Fill the slurm job file
    with open(job_file, "w+") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -J P16panc_inference\n")
        fh.writelines("#SBATCH -p LKEBgpu\n")
        fh.writelines("#SBATCH -N 1\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines(f"#SBATCH --cpus-per-task={cpu}\n")
        fh.writelines("#SBATCH --time=02:00:00\n")
        fh.writelines("#SBATCH --mem=85GB\n")
        fh.writelines("#SBATCH --gres=gpu:RTX6000:1\n")
        fh.writelines(f"#SBATCH --error={env_vars['log_dir']}/job.%J.inference.err\n")
        fh.writelines(f"#SBATCH --output={env_vars['log_dir']}/job.%J.inference.out\n\n")
        
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
        fh.writelines("echo \"Activate conda env nnUNet\"\n")
        fh.writelines(f"source {env_vars['conda_env_dir']}\n")
        fh.writelines("echo \"Verifying environment variables:\"\n")
        fh.writelines("conda env config vars list\n")
        fh.writelines("echo \"Installing hidden layer and nnUnet\"\n\n")
        fh.writelines("python -m pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git@more_plotted_details#egg=hiddenlayer\n")
        fh.writelines(f"python -m pip install --editable {env_vars['nnUNet_code_dir']}\n")

        fh.writelines(f"mkdir -p $nnUNet_raw_data_base/p16/batch{batch}/segmentations/Task{task}\n")
        fh.writelines(f"nnUNet_predict -i $nnUNet_raw_data_base/p16/batch{batch}/niftis -o $nnUNet_raw_data_base/p16/batch{batch}/segmentations/Task{task}  -t {task} -m 3d_fullres -tr nnUNetTrainerV2 -f 0 1 2 3 4\n")
        fh.writelines("echo \"Program finished with exit code $? at: `\date`\"")
        fh.writelines("")
    
    # Submit batch file
    os.system("sbatch %s" %job_file)
    
