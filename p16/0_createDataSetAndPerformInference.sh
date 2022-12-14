# Load the environment variables
echo ""
echo "Load the environment variables:"
source ENVIRONMENT_VARIABLES.txt
echo "job_dir is $job_dir"
echo "log_dir is $log_dir"
echo "conda_env_dir is $conda_env_dir"
echo "nnunet_code_dir is $nnUNet_code_dir"
echo ""

# Activate the conda environment
echo "Activate the conda environment"
echo ""
source $conda_env_dir

# Create the dataset
echo "Create a new nnU-Net dataset from the new batch."
python scripts/createP16Dataset.py $pwd

# Perform inference
echo "Use the latest model to generate segmentations for the new batch."
python scripts/runInference.py

