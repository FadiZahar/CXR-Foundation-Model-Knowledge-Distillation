# Configuration for MIMIC-CXR settings

## Knowledge Distillation
BEST_CHECKPOINT_KD_MSE_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-MSE_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSE_epoch=39-val_loss=0.0810.ckpt'
BEST_CHECKPOINT_KD_MAE_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-MAE_multirun-seed42/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MAE_epoch=39-val_loss=0.2214.ckpt'
BEST_CHECKPOINT_KD_HuberLoss_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-HuberLoss_multirun-seed43/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_HuberLoss_epoch=39-val_loss=0.0404.ckpt'
BEST_CHECKPOINT_KD_CosineSim_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-CosineSim_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_CosineSim_epoch=39-val_loss=0.0099.ckpt'
BEST_CHECKPOINT_KD_MSEandCosineSim_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-MSEandCosineSim_multirun-seed42/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSim_epoch=39-val_loss=0.0453.ckpt'
BEST_CHECKPOINT_KD_MSEandCosineSimLearned_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-MSEandCosineSimLearned_multirun-seed41/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimLearned_epoch=39-val_loss=-5.0866.ckpt'
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p50_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-MSEandCosineSimWeighted-alpha0p50_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p50_epoch=39_val_loss=1.1227.ckpt' # alpha = 0.5
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p60_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-MSEandCosineSimWeighted-alpha0p60_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p60_epoch=39_val_loss=1.1123.ckpt' # alpha = 0.6
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p70_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-MSEandCosineSimWeighted-alpha0p70_multirun-seed44/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p70_epoch=39_val_loss=1.0894.ckpt' # alpha = 0.7
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p80_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-MSEandCosineSimWeighted-alpha0p80_multirun-seed43/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p80_epoch=39_val_loss=1.0822.ckpt' # alpha = 0.8
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p90_FILENAME = 'multiruns/MIMIC_CXR-FMKD_KD-initialisation-MSEandCosineSimWeighted-alpha0p90_multirun-seed43/lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p90_epoch=39_val_loss=1.0792.ckpt' # alpha = 0.9

## Models
BEST_CHECKPOINT__CXR_FM_linear_probing__FILEPATH = ''
BEST_CHECKPOINT__CXR_model_full_finetuning__FILEPATH = ''
BEST_CHECKPOINT__CXR_model_linear_probing__FILEPATH = ''
# MSE:
ORIGINAL_MSE_KD_TYPE_DIR_NAME = 'KD-MSE'
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__MSE__FILEPATH = ''
BEST_CHECKPOINT__CXR_FMKD_linear_probing__MSE__FILEPATH = ''
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__MSE__FILEPATH = ''
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__MSE__FILEPATH = ''
# CS:
ORIGINAL_CS_KD_TYPE_DIR_NAME = 'KD-CosineSim'
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__CS__FILEPATH = ''
BEST_CHECKPOINT__CXR_FMKD_linear_probing__CS__FILEPATH = ''
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__CS__FILEPATH = ''
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__CS__FILEPATH = ''
# MSE&CS Combination:
ORIGINAL_MSEandCS_KD_TYPE_DIR_NAME = 'KD-MSEandCosineSimWeighted-alpha0p90'
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__MSEandCS__FILEPATH = ''
BEST_CHECKPOINT__CXR_FMKD_linear_probing__MSEandCS__FILEPATH = ''
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__MSEandCS__FILEPATH = ''
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__MSEandCS__FILEPATH = ''

## Data Paths
CXRS_FILEPATH = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/'
EMBEDDINGS_FILEPATH = '/vol/biomedic3/bglocker/mscproj24/fz221/data/cxrfm_embeddings/mimic/cxr_numpy'
TRAIN_RECORDS_CSV = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/meta/algorithmic_encoding/mimic.sample.train.csv'
VAL_RECORDS_CSV = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/meta/algorithmic_encoding/mimic.sample.val.csv'
TEST_RECORDS_CSV = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/meta/algorithmic_encoding/mimic.resample.test.csv'  # Using resampled test set
MAIN_DIR_PATH = '/vol/biomedic3/bglocker/mscproj24/fz221/outputs/outputs_mimic/'
INFER_DIR_PATH = '/vol/biomedic3/bglocker/mscproj24/fz221/inference/inference_on_mimic/'
