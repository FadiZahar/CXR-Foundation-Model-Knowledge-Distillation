# Configuration for CheXpert settings

## Knowledge Distillation
BEST_CHECKPOINT_KD_MSE_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-MSE_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSE_epoch=38-val_loss=0.0940.ckpt'
BEST_CHECKPOINT_KD_MAE_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-MAE_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MAE_epoch=39-val_loss=0.2395.ckpt'
BEST_CHECKPOINT_KD_HuberLoss_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-HuberLoss_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_HuberLoss_epoch=38-val_loss=0.0469.ckpt'
BEST_CHECKPOINT_KD_CosineSim_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-CosineSim_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_CosineSim_epoch=38-val_loss=0.0106.ckpt'
BEST_CHECKPOINT_KD_MSEandCosineSim_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-MSEandCosineSim_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSim_epoch=39-val_loss=0.0523.ckpt'
BEST_CHECKPOINT_KD_MSEandCosineSimLearned_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-MSEandCosineSimLearned_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimLearned_epoch=38-val_loss=-4.8428.ckpt'
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p50_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-MSEandCosineSimWeighted-alpha0p50_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p50_epoch=38_val_loss=1.2414.ckpt' # alpha = 0.5
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p60_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-MSEandCosineSimWeighted-alpha0p60_multirun-seed42/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p60_epoch=39_val_loss=1.2266.ckpt' # alpha = 0.6
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p70_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-MSEandCosineSimWeighted-alpha0p70_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p70_epoch=38_val_loss=1.2395.ckpt' # alpha = 0.7
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p80_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-MSEandCosineSimWeighted-alpha0p80_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p80_epoch=38_val_loss=1.2419.ckpt' # alpha = 0.8
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p90_FILENAME = 'multiruns/CheXpert_CXR-FMKD_KD-initialisation-MSEandCosineSimWeighted-alpha0p90_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p90_epoch=38_val_loss=1.2418.ckpt' # alpha = 0.9

## Models
BEST_CHECKPOINT__CXR_FM_linear_probing__FILEPATH = 'multiruns/CheXpert_CXR-FM_linear-probing_multirun-seed45/lightning_checkpoints/best-checkpoint_CXR-FM_lp_epoch=11-val_loss=0.3005.ckpt'
BEST_CHECKPOINT__CXR_model_full_finetuning__FILEPATH = 'multiruns/CheXpert_CXR-model_full-finetuning_multirun-seed44/lightning_checkpoints/best-checkpoint_CheXpert-model_fft_epoch=14-val_loss=0.2862.ckpt'
BEST_CHECKPOINT__CXR_model_linear_probing__FILEPATH = 'multiruns/CheXpert_CXR-model_linear-probing_multirun-seed42/lightning_checkpoints/best-checkpoint_CheXpert-model_lp_epoch=10-val_loss=0.3206.ckpt'
# MSE:
ORIGINAL_MSE_KD_TYPE_DIR_NAME = 'KD-MSE'
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__MSE__FILEPATH = 'multiruns/CheXpert_CXR-FMKD_full-finetuning_multirun-seed43/lightning_checkpoints/best-checkpoint_CXR-FMKD_fft_epoch=1-val_loss=0.2829.ckpt'
BEST_CHECKPOINT__CXR_FMKD_linear_probing__MSE__FILEPATH = 'multiruns/CheXpert_CXR-FMKD_linear-probing_multirun-seed43/lightning_checkpoints/best-checkpoint_CXR-FMKD_lp_epoch=19-val_loss=0.2970.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__MSE__FILEPATH = 'multiruns/CheXpert_CXR-FMKD-1664to14_full-finetuning_multirun-seed42/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_fft_epoch=2-val_loss=0.2781.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__MSE__FILEPATH = 'multiruns/CheXpert_CXR-FMKD-1664to14_linear-probing_multirun-seed42/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_lp_epoch=18-val_loss=0.2911.ckpt'
# CS:
ORIGINAL_CS_KD_TYPE_DIR_NAME = 'KD-CosineSim'
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__CS__FILEPATH = 'multiruns/CheXpert_CXR-FMKD_full-finetuning_multirun-seed43/lightning_checkpoints/best-checkpoint_CXR-FMKD_fft_epoch=7-val_loss=0.2847.ckpt'
BEST_CHECKPOINT__CXR_FMKD_linear_probing__CS__FILEPATH = 'multiruns/CheXpert_CXR-FMKD_linear-probing_multirun-seed44/lightning_checkpoints/best-checkpoint_CXR-FMKD_lp_epoch=2-val_loss=3.5920.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__CS__FILEPATH = 'multiruns/CheXpert_CXR-FMKD-1664to14_full-finetuning_multirun-seed45/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_fft_epoch=1-val_loss=0.2781.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__CS__FILEPATH = 'multiruns/CheXpert_CXR-FMKD-1664to14_linear-probing_multirun-seed43/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_lp_epoch=3-val_loss=0.2901.ckpt'
# MSE&CS Combination:
ORIGINAL_MSEandCS_KD_TYPE_DIR_NAME = 'KD-MSEandCosineSimWeighted-alpha0p60'
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__MSEandCS__FILEPATH = 'multiruns/CheXpert_CXR-FMKD_full-finetuning_multirun-seed41/lightning_checkpoints/best-checkpoint_CXR-FMKD_fft_epoch=2-val_loss=0.2807.ckpt'
BEST_CHECKPOINT__CXR_FMKD_linear_probing__MSEandCS__FILEPATH = 'multiruns/CheXpert_CXR-FMKD_linear-probing_multirun-seed43/lightning_checkpoints/best-checkpoint_CXR-FMKD_lp_epoch=19-val_loss=0.2968.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__MSEandCS__FILEPATH = 'multiruns/CheXpert_CXR-FMKD-1664to14_full-finetuning_multirun-seed41/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_fft_epoch=3-val_loss=0.2768.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__MSEandCS__FILEPATH = 'multiruns/CheXpert_CXR-FMKD-1664to14_linear-probing_multirun-seed43/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_lp_epoch=19-val_loss=0.2907.ckpt'

## Data Paths
CXRS_FILEPATH = '/vol/biodata/data/chest_xray/CheXpert-v1.0/'
EMBEDDINGS_FILEPATH = '/vol/biomedic3/bglocker/mscproj24/fz221/data/cxrfm_embeddings/chexpert/cxr_numpy'
TRAIN_RECORDS_CSV = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.train.csv'
VAL_RECORDS_CSV = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.val.csv'
TEST_RECORDS_CSV = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.resample.test.csv'  # Using resampled test set
MAIN_DIR_PATH = '/vol/biomedic3/bglocker/mscproj24/fz221/outputs/outputs_chexpert/'
INFER_DIR_PATH = '/vol/biomedic3/bglocker/mscproj24/fz221/inference/inference_on_chexpert/'
