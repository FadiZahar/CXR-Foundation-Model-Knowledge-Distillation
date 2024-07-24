#!/bin/bash
#SBATCH --partition=gpus24       
#SBATCH --gres=gpu:1             
#SBATCH --output=/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/baselines/slurm_scripts/cxrfm_lp.%N.%j.log  
#SBATCH --time=3-00:00:00        

#SBATCH --job-name=cxrfm_lp

source /vol/bitbucket/fz221/fmkd_venv/bin/activate
export PYTHONPATH=/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/

python /vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/baselines/disease_prediction__CXR_FM__linear_probing.py
