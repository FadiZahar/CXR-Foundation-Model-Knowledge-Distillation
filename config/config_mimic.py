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
BEST_CHECKPOINT__CXR_FM_linear_probing__FILEPATH = 'multiruns/MIMIC_CXR-FM_linear-probing_multirun-seed43/lightning_checkpoints/best-checkpoint_CXR-FM_lp_epoch=7-val_loss=0.2826.ckpt'
BEST_CHECKPOINT__CXR_model_full_finetuning__FILEPATH = 'multiruns/MIMIC_CXR-model_full-finetuning_multirun-seed45/lightning_checkpoints/best-checkpoint_CXR-model_fft_epoch=9-val_loss=0.2644.ckpt'
BEST_CHECKPOINT__CXR_model_linear_probing__FILEPATH = 'multiruns/MIMIC_CXR-model_linear-probing_multirun-seed41/lightning_checkpoints/best-checkpoint_CXR-model_lp_epoch=16-val_loss=0.2967.ckpt'
# MSE:
ORIGINAL_MSE_KD_TYPE_DIR_NAME = 'KD-MSE'
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__MSE__FILEPATH = 'multiruns/MIMIC_CXR-FMKD_full-finetuning_multirun-seed42/lightning_checkpoints/best-checkpoint_CXR-FMKD_fft_epoch=1-val_loss=0.2632.ckpt'
BEST_CHECKPOINT__CXR_FMKD_linear_probing__MSE__FILEPATH = 'multiruns/MIMIC_CXR-FMKD_linear-probing_multirun-seed41/lightning_checkpoints/best-checkpoint_CXR-FMKD_lp_epoch=9-val_loss=0.2784.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__MSE__FILEPATH = 'multiruns/MIMIC_CXR-FMKD-1664to14_full-finetuning_multirun-seed42/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_fft_epoch=4-val_loss=0.2625.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__MSE__FILEPATH = 'multiruns/MIMIC_CXR-FMKD-1664to14_linear-probing_multirun-seed41/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_lp_epoch=19-val_loss=0.2737.ckpt'
# CS:
ORIGINAL_CS_KD_TYPE_DIR_NAME = 'KD-CosineSim'
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__CS__FILEPATH = 'multiruns/MIMIC_CXR-FMKD_full-finetuning_multirun-seed45/lightning_checkpoints/best-checkpoint_CXR-FMKD_fft_epoch=5-val_loss=0.2665.ckpt'
BEST_CHECKPOINT__CXR_FMKD_linear_probing__CS__FILEPATH = 'multiruns/MIMIC_CXR-FMKD_linear-probing_multirun-seed42/lightning_checkpoints/best-checkpoint_CXR-FMKD_lp_epoch=15-val_loss=9.6979.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__CS__FILEPATH = 'multiruns/MIMIC_CXR-FMKD-1664to14_full-finetuning_multirun-seed45/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_fft_epoch=1-val_loss=0.2614.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__CS__FILEPATH = 'multiruns/MIMIC_CXR-FMKD-1664to14_linear-probing_multirun-seed41/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_lp_epoch=6-val_loss=0.2730.ckpt'
# MSE&CS Combination:
ORIGINAL_MSEandCS_KD_TYPE_DIR_NAME = 'KD-MSEandCosineSimWeighted-alpha0p90'
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__MSEandCS__FILEPATH = 'multiruns/MIMIC_CXR-FMKD_full-finetuning_multirun-seed41/lightning_checkpoints/best-checkpoint_CXR-FMKD_fft_epoch=2-val_loss=0.2651.ckpt'
BEST_CHECKPOINT__CXR_FMKD_linear_probing__MSEandCS__FILEPATH = 'multiruns/MIMIC_CXR-FMKD_linear-probing_multirun-seed43/lightning_checkpoints/best-checkpoint_CXR-FMKD_lp_epoch=8-val_loss=0.2770.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__MSEandCS__FILEPATH = 'multiruns/MIMIC_CXR-FMKD-1664to14_full-finetuning_multirun-seed45/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_fft_epoch=3-val_loss=0.2626.ckpt'
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__MSEandCS__FILEPATH = 'multiruns/MIMIC_CXR-FMKD-1664to14_linear-probing_multirun-seed42/lightning_checkpoints/best-checkpoint_CXR-FMKD-1664to14_lp_epoch=19-val_loss=0.2734.ckpt'

## Data Paths
CXRS_FILEPATH = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/'
EMBEDDINGS_FILEPATH = '/vol/biomedic3/bglocker/mscproj24/fz221/data/cxrfm_embeddings/mimic/cxr_numpy'
TRAIN_RECORDS_CSV = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/meta/algorithmic_encoding/mimic.sample.train.csv'
VAL_RECORDS_CSV = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/meta/algorithmic_encoding/mimic.sample.val.csv'
TEST_RECORDS_CSV = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/meta/algorithmic_encoding/mimic.resample.test.csv'  # Using resampled test set
MAIN_DIR_PATH = '/vol/biomedic3/bglocker/mscproj24/fz221/outputs/outputs_mimic/'
INFER_DIR_PATH = '/vol/biomedic3/bglocker/mscproj24/fz221/inference/inference_on_mimic/'
