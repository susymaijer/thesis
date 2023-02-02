#!/usr/bin/python
import os
import sys
from datetime import date, datetime
# custom module
import shared

'''
    This script evaluates the performance of the most recent model, by comparing the predictions with the corrections from the medical exper.
'''

if __name__ == "__main__":

    # Get the batch and task
    batch = shared.determine_batch_id()
    task = shared.determine_task_id()-1
    print(f"We're going to evaluate the corrections on batch {batch}. Is this OK? Enter [y/n].")
    answer = input()
    if answer != "y":
        print("Please enter the batch id.")
        batch = input()    
    print(f"We're going to evaluate the predictions made by model of task{task}. Is this OK? Enter [y/n].")
    answer = input()
    if answer != "y":
        print("Please enter the task id.")
        task = input()

    # Get cpu amount
    print(f"How many cpu's do you want to use? Recommended values are [4,6,8].")
    cpu = input()

    # Get the environment variables
    if len(sys.argv) > 1:
        env_var_dir = sys.argv[1]
    else:
        env_var_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    env_vars = shared.getUserspecificEnvironmentVariables(os.path.join(env_var_dir, "ENVIRONMENT_VARIABLES.txt"))

    # Define the paths to the folder containing the niftis which we are going to predict
    batch_dir = os.path.join(env_vars['p16_dir'], f'batch{batch}')
    corr_dir = os.path.join(batch_dir, 'corr')
    auto_dir = os.path.join(batch_dir, 'segmentations', f"Task{task}")

    # Get slurm job file path
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    job_file = os.path.join(env_vars['job_dir'], f"p16_evaluate_batch{batch}_{ts}.job")
    
    # Fill the slurm job file
    with open(job_file, "w+") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -J P16panc_evaluate\n")
        fh.writelines("#SBATCH -p LKEBgpu\n")
        fh.writelines("#SBATCH -N 1\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines(f"#SBATCH --cpus-per-task={cpu}\n")
        fh.writelines("#SBATCH --time=02:00:00\n")
        fh.writelines("#SBATCH --mem=85GB\n")
        fh.writelines("#SBATCH --gres=gpu:RTX6000:1\n")
        fh.writelines(f"#SBATCH --error={env_vars['log_dir']}/job.%J.evalute.err\n")
        fh.writelines(f"#SBATCH --output={env_vars['log_dir']}/job.%J.evaluate.out\n\n")
        
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

        fh.writelines(f"nnUNet_evaluate_folder -ref {corr_dir} -pred {auto_dir} -l 1\n")
        fh.writelines("echo \"Program finished with exit code $? at: `\date`\"")
        fh.writelines("")
    
    # Submit batch file
    os.system("sbatch %s" %job_file)
    
