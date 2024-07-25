#!/bin/bash
#SBATCH --partition=gpus24       
#SBATCH --gres=gpu:1             
#SBATCH --output=/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/models/slurm_scripts/resnet50_fft.%N.%j.log  
#SBATCH --time=3-00:00:00        

#SBATCH --job-name=resnet50_fft

source /vol/bitbucket/fz221/fmkd_venv/bin/activate
export PYTHONPATH=/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/

python /vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/models/disease_prediction__ResNet50__full_finetuning.py
