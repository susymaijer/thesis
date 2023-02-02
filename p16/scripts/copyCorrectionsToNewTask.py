#!/usr/bin/python
import os
import sys
import shutil
from pydicom import dcmread 
from pydicom.fileset import FileSet
import shared # our own

''' 
    Copy the corrections of the predictions made by the radiologist to labelsTr folder of the most recent P16 task.
'''

if __name__ == "__main__":

    # Get the batch id and task id
    batch = shared.determine_batch_id()
    task = shared.determine_task_id() 
    print(f"We're going to copy corrections from batch {batch} to task {task}. Is this OK? Enter [y/n].")
    answer = input()
    if answer != "y":
        sys.exit("Abort")

    # Get the environment variables
    if len(sys.argv) > 1:
        env_var_dir = sys.argv[1]
    else:
        env_var_dir = os.path.dirname(os.path.dirname((os.path.abspath(__file__))))
    env_vars = shared.getUserspecificEnvironmentVariables(os.path.join(env_var_dir, "ENVIRONMENT_VARIABLES.txt"))

    # Define the paths to the folder containing the scans which we want to add the ground truths for
    corr_labels_dir = os.path.join(env_vars['p16_dir'], f'batch{batch}', 'corr')
    labels_dir = os.path.join(os.environ.get('nnUNet_raw_data_base'), 'nnUNet_raw_data', f'Task{task}', 'labelsTr')

    # Copy contents of previous task to new task
    if (not(os.path.exists(labels_dir))):
        sys.exit(f"Directory {labels_dir} does not exist. Abort program.")
    else:
        # Add the corrected ground truth to LabelsTr of most recent task
        for file_name in os.listdir(corr_labels_dir):
            shutil.copy(os.path.join(corr_labels_dir,file_name), os.path.join(labels_dir,file_name))
            print('copied', file_name)
        print(f"Copied contents of {corr_labels_dir} to {labels_dir}\n")
    
