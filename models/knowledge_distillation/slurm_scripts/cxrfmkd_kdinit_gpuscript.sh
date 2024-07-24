#!/bin/bash
#SBATCH --partition=gpus24       
#SBATCH --gres=gpu:1             
#SBATCH --output=/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/models/knowledge_distillation/slurm_scripts/cxrfmkd_kdinit.%N.%j.log  
#SBATCH --time=3-00:00:00        

#SBATCH --job-name=cxrfmkd_kdinit

source /vol/bitbucket/fz221/fmkd_venv/bin/activate
export PYTHONPATH=/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/

python /vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/models/knowledge_distillation__CXR_FMKD__initialisation.py
