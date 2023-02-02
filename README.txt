# Use the following command to set conda environment variables:
conda env config vars set X="Y"

# For all the following variables:
nnUNet_raw_data_base="/exports/lkeb-hpc/smaijer/data/nnUNet_raw_data_base"
nnUNet_preprocessed="/exports/lkeb-hpc/smaijer/data/nnUNet_preprocessed"
RESULTS_FOLDER="/exports/lkeb-hpc/smaijer/results"
OUTPUT="/exports/lkeb-hpc/smaijer/output"

# Eenmalig
python3 -m venv pancreas
/share/software/system/python/3.10.2/bin/python3.10 -m pip install --upgrade pip
pip config set pancreas.target /exports/lkeb-hpc/smaijer/venv_environments/pancreas
