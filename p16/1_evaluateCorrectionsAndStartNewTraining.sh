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

# Run evaluation of the corrected scans
echo "Run evaluation of the corrected scans"
#python scripts/runEvaluation.py

# Copy corrections to new task
echo "Copy corrections to new task"
#python scripts/copyCorrectionsToNewTask.py

# Start new training run, using the weights of the previous model as the starting weights
echo "Start new training run, using the weights of the previous model as the starting weights"
python scripts/runTraining.py

