#!/bin/bash
#SBATCH --partition=gpus48       
#SBATCH --gres=gpu:1             
#SBATCH --output=/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/models/slurm_scripts/slurm_logs/cxrfmkd/cxrfmkd_mse&cosinesimweighted_alpha0p70/%x.%N.%j.log  
#SBATCH --time=0-05:00:00 
#SBATCH --nodelist=loki 

#SBATCH --job-name=cxrfmkd1664to14_lp_mse&cosinesimweighted_alpha0p70

# Activate the Python virtual environment (fmkd_venv)
source /vol/bitbucket/fz221/fmkd_venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/:$PYTHONPATH

# Set wandb directories
export WANDB_CACHE_DIR=/vol/biomedic3/bglocker/mscproj24/fz221/.wandb_storage/cache
export WANDB_CONFIG_DIR=/vol/biomedic3/bglocker/mscproj24/fz221/.wandb_storage/config

# Run the Python script
python /vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/models/disease_prediction__CXR_FMKD_1664to14__linear_probing.py --kd_type 'MSEandCosineSimWeighted-alpha0p70' --config "$1"
