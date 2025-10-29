#!/bin/bash
#SBATCH --job-name=setup_conda_env
#SBATCH --time=01:00:00
#SBATCH --partition=devel
#SBATCH --account=<your-project>   # e.g. jsc1234
#SBATCH --output=setup_env.log

# Load base Python module
module load Python

# Paths (edit as needed)
PROJECT_DIR=$PROJECT/power-system-split
CONDA_DIR=$PROJECT_DIR/conda
ENV_FILE=$PROJECT_DIR/environment.yml
ENV_PATH=$CONDA_DIR/envs/myenv

# Install Miniforge if not yet installed
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing Miniforge..."
    wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniforge3.sh
    bash Miniforge3.sh -b -p $CONDA_DIR
    rm Miniforge3.sh
fi

# Configure Conda
$CONDA_DIR/bin/conda config --add channels conda-forge
$CONDA_DIR/bin/conda config --set channel_priority strict
$CONDA_DIR/bin/conda config --set auto_activate_base false

# Create environment from YAML
echo "Creating environment from $ENV_FILE..."
$CONDA_DIR/bin/conda env create -f $ENV_FILE -p $ENV_PATH

# Verify installation
source $CONDA_DIR/bin/activate $ENV_PATH
python --version
conda list

echo "Environment setup complete: $ENV_PATH"
