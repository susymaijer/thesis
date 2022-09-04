#!/bin/bash

echo "Convert all dicom folders in a specific folder to niifti's".
echo "" 

read -p "Enter folder name:" folder

echo "Load python modules.."
module add system/python/3.10.2
echo "Activate environment.."
source /exports/lkeb-hpc/smaijer/venv_environments/susy/bin/activate

python preprocessing/convertDicomToNifti.py $folder

