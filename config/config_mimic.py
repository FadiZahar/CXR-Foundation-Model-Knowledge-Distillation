# Configuration for MIMIC-CXR settings

# Knowledge Distillation
BEST_CHECKPOINT_KD_MSE_FILENAME = ''
BEST_CHECKPOINT_KD_MAE_FILENAME = ''
BEST_CHECKPOINT_KD_HuberLoss_FILENAME = ''
BEST_CHECKPOINT_KD_CosineSim_FILENAME = ''
BEST_CHECKPOINT_KD_MSEandCosineSim_FILENAME = ''
BEST_CHECKPOINT_KD_MSEandCosineSimLearned_FILENAME = ''
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p50_FILENAME = '' # alpha = 0.5
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p60_FILENAME = '' # alpha = 0.6
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p70_FILENAME = '' # alpha = 0.7
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p80_FILENAME = '' # alpha = 0.8
BEST_CHECKPOINT_KD_MSEandCosineSimWeighted_alpha0p90_FILENAME = '' # alpha = 0.9

# Models
BEST_CHECKPOINT__CXR_FM_linear_probing__FILENAME = ''
BEST_CHECKPOINT__CXR_FMKD_full_finetuning__FILENAME = ''
BEST_CHECKPOINT__CXR_FMKD_linear_probing__FILENAME = ''
BEST_CHECKPOINT__CXR_FMKD_1664to14_full_finetuning__FILENAME = ''
BEST_CHECKPOINT__CXR_FMKD_1664to14_linear_probing__FILENAME = ''
BEST_CHECKPOINT__CXR_model_full_finetuning__FILENAME = ''
BEST_CHECKPOINT__CXR_model_linear_probing__FILENAME = ''

# Data Paths
CXRS_FILEPATH = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/data/'
EMBEDDINGS_FILEPATH = '/vol/biomedic3/bglocker/mscproj24/fz221/data/cxrfm_embeddings/mimic/cxr_numpy'
TRAIN_RECORDS_CSV = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/meta/algorithmic_encoding/mimic.sample.train.csv'
VAL_RECORDS_CSV = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/meta/algorithmic_encoding/mimic.sample.val.csv'
TEST_RECORDS_CSV = '/vol/biodata/data/chest_xray/mimic-cxr-jpg-224/meta/algorithmic_encoding/mimic.resample.test.csv'  # Using resampled test set
MAIN_DIR_PATH = '/vol/biomedic3/bglocker/mscproj24/fz221/outputs/outputs_mimic/'
INFER_DIR_PATH = '/vol/biomedic3/bglocker/mscproj24/fz221/inference/inference_on_mimic/'
