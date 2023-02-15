# Use the following command to set conda environment variables:
conda env config vars set X="Y"

# Set all the necessary nnU-Net variables:
nnUNet_raw_data_base="/exports/lkeb-hpc/smaijer/data/nnUNet_raw_data_base"
nnUNet_preprocessed="/exports/lkeb-hpc/smaijer/data/nnUNet_preprocessed"
RESULTS_FOLDER="/exports/lkeb-hpc/smaijer/results"

# Set a variable we want to use ourselves
OUTPUT="/exports/lkeb-hpc/smaijer/output"

# Set up a virtual environment using venv
module purge
module add system/python/3.10.2
python3 -m venv pancreas
/share/software/system/python/3.10.2/bin/python3.10 -m pip install --upgrade pip

# On default, environments get put in your home directory
# We don't want this, since environments can take up a lot of space and there is a hard max of 10GB in your home directory
# So specify another location where there is not space limit
mkdir /exports/lkeb-hpc/smaijer/venv_environments
pip config set pancreas.target /exports/lkeb-hpc/smaijer/venv_environments/pancreas

echo "Installing nnU-net.."
source /exports/lkeb-hpc/smaijer/venv_environments/pancreasThesis/bin/activate
python pip install --upgrade --force-reinstal torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'
python -m pip install --editable /home/smaijer/code/nnUNet
# PREFERRED but I had troubles so I did the above
# pip install -e /home/smaijer/code/nnUNet

echo "Verify pytorch"
python -c 'import torch;print(torch.backends.cudnn.version())'
python -c 'import torch;print(torch.__version__)'

FURTHER:
-tasks starting with 61 XX are P16 tasks. But please correct this to 500.
-copy my p16 data to your own folder
-copy the most recent models (from $RESULTS and $preprocessed) to your own folder

ALSO NOTE:
-always check identify.txt if there are not extra scans that should not be there
-also. nnU-Net offers loads of options, like disable TTA or singlethreading workers. I recommend reading the repository thorougly.

