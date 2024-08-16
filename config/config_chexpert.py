# Configuration for CheXpert settings

# Knowledge Distillation
BEST_CHECKPOINT_KD_MSE_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSE_epoch=38-val_loss=0.0940.ckpt'
BEST_CHECKPOINT_KD_MAE_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MAE_epoch=39-val_loss=0.2397.ckpt'
BEST_CHECKPOINT_KD_HuberLoss_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_HuberLoss_epoch=39-val_loss=0.0468.ckpt'
BEST_CHECKPOINT_KD_CosineSim_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_CosineSim_epoch=38-val_loss=0.0106.ckpt'
BEST_CHECKPOINT_KD_MSEandCosineSim_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSim_epoch=38-val_loss=0.0521.ckpt'
BEST_CHECKPOINT_KD_MSEandCosineSimLearned_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimLearned_epoch=37-val_loss=-4.8417.ckpt'
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p50_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p50_epoch=38_val_loss=1.3288.ckpt' # alpha = 0.5
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p60_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p60_epoch=39_val_loss=1.2477.ckpt' # alpha = 0.6
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p70_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p70_epoch=38_val_loss=1.2580.ckpt' # alpha = 0.7
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p80_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p80_epoch=39_val_loss=1.2486.ckpt' # alpha = 0.8
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p90_FILENAME = 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD_MSEandCosineSimWeighted-alpha0p90_epoch=39_val_loss=1.2538.ckpt' # alpha = 0.9

# Models
BEST_CHECKPOINT__CXR_FM_linear_probing__FILENAME = ''
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__FILENAME = ''
BEST_CHECKPOINT__CXR_FMKD_linear_probing__FILENAME = ''
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__FILENAME = ''
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__FILENAME = ''
BEST_CHECKPOINT__CXR_model_full_finetuning__FILENAME = ''
BEST_CHECKPOINT__CXR_model_linear_probing__FILENAME = ''

# Data Paths
CXRS_FILEPATH = '/vol/biodata/data/chest_xray/CheXpert-v1.0/'
EMBEDDINGS_FILEPATH = '/vol/biomedic3/bglocker/mscproj24/fz221/data/cxrfm_embeddings/chexpert/cxr_numpy'
TRAIN_RECORDS_CSV = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.train.csv'
VAL_RECORDS_CSV = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.val.csv'
TEST_RECORDS_CSV = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.resample.test.csv'  # Using resampled test set
MAIN_DIR_PATH = '/vol/biomedic3/bglocker/mscproj24/fz221/outputs/outputs_chexpert/'
INFER_DIR_PATH = '/vol/biomedic3/bglocker/mscproj24/fz221/inference/inference_on_chexpert/'
