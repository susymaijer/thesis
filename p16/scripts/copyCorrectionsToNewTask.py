#!/usr/bin/python
import os
import sys
import glob
import shutil
import random
import zipfile
from datetime import date
from collections import OrderedDict
import SimpleITK as sitk
from pydicom import dcmread 
from pydicom.fileset import FileSet
import shared # our own

''' 

        TODO
'''

if __name__ == "__main__":

    # Get the batch id and task id
    batch=shared.determine_batch_id()
    task=shared.determine_task_id() 
    print(f"We're going to copy corrections from batch {batch} to task {task}. Is this OK? Enter [y/n].")
    answer=input()
    if answer != "y":
        sys.exit("Abort")

    # Get the environment variables
    if len(sys.argv) > 1:
        env_var_dir=sys.argv[1]
    else:
        env_var_dir=os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    env_vars=shared.getUserspecificEnvironmentVariables(os.path.join(env_var_dir, "ENVIRONMENT_VARIABLES.txt"))

    # Define the paths to the folder containing the scans which we need to convert 
    img_dir=os.path.join(env_vars['p16_dir'], f'batch{batch}', 'corr')
    labels=os.path.join(os.environ.get('nnUNet_raw_data_base'), 'nnUNet_raw_data', f'Task{task}', 'labelsTr')

    # Copy contents of previous task to new task
    if (not(os.path.exists(labels_dir))):
        sys.exit(f"Directory {labels_dir} does not exist. Abort program.")
    else:
        # Add the data of the old task to the new directory
        shutil.copytree(prev_task_dir, out_dir)
        print(f"Copied contents of {prev_task_dir} to new task dir\n")
    
