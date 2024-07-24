#!/bin/bash
#SBATCH --partition=gpus   
#SBATCH --gres=gpu:1             
#SBATCH --output=/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/models/slurm_scripts/chexmod_fft.%N.%j.log  
#SBATCH --time=3-00:00:00
#SBATCH --nodelist=monal03        

#SBATCH --job-name=chexmod_fft

source /vol/bitbucket/fz221/fmkd_venv/bin/activate
export PYTHONPATH=/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/

python /vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/models/disease_prediction__CheXpert_model__full_finetuning.py
