import os
import numpy as np
from skimage.io import imsave
from argparse import ArgumentParser
from datetime import datetime

import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


# Import custom modules
from data_modules.cxr_data_module import CXRDataModule
from utils.output_utils.generate_and_save_raw_outputs import run_evaluation_phase
from utils.output_utils.generate_and_save_metrics import save_and_plot_all_metrics

# Import global variables
from config.config_shared import IMAGE_SIZE, NUM_CLASSES, EPOCHS, NUM_WORKERS, BATCH_SIZE, LEARNING_RATE, TARGET_FPR
# Import the configuration loader
from config.loader_config import load_config, get_dataset_name

# Model import
from models.disease_prediction__CXR_model__linear_probing import CXRModel_LinearProbing

pre_OUT_DIR_NAME = 'CXR-model_linear-probing/'



def main(hparams):

    # Decide which dataset's configuration to load for test data and which dataset's checkpoint to use
    if hparams.inference_on == 'mimic':
        test_config = load_config('mimic')
        model_config = load_config('chexpert')
    else:
        test_config = load_config('chexpert')
        model_config = load_config('mimic')

    # From test config:
    CXRS_FILEPATH = test_config.CXRS_FILEPATH
    EMBEDDINGS_FILEPATH = test_config.EMBEDDINGS_FILEPATH
    TRAIN_RECORDS_CSV = test_config.TRAIN_RECORDS_CSV
    VAL_RECORDS_CSV = test_config.VAL_RECORDS_CSV
    TEST_RECORDS_CSV = test_config.TEST_RECORDS_CSV
    INFER_DIR_PATH = test_config.INFER_DIR_PATH
    # From model config:
    MAIN_DIR_PATH = model_config.MAIN_DIR_PATH
    BEST_CHECKPOINT_FILEPATH = model_config.BEST_CHECKPOINT__CXR_model_linear_probing__FILEPATH


    # Updated OUT_DIR_NAME to include dataset name
    dataset_name = get_dataset_name(hparams.inference_on)
    prev_OUT_DIR_NAME = dataset_name + '_' + pre_OUT_DIR_NAME
    OUT_DIR_NAME = 'ZSInfer_on_' + prev_OUT_DIR_NAME

    # Get model checkpiont full path
    BEST_CHECKPOINT_FULLPATH = os.path.join(MAIN_DIR_PATH, prev_OUT_DIR_NAME, BEST_CHECKPOINT_FILEPATH)


    # Create output directory
    if hparams.multirun_seed:
        inner_out_dir_name = f"{OUT_DIR_NAME.strip('/')}_multirun-seed{hparams.multirun_seed}"
        out_dir_path = os.path.join(INFER_DIR_PATH, 'ZSInfer', OUT_DIR_NAME, 'multiruns', inner_out_dir_name)
    else:
        out_dir_path = os.path.join(INFER_DIR_PATH, 'ZSInfer', OUT_DIR_NAME)
    os.makedirs(out_dir_path, exist_ok=True)
    # Create TensorBoard logs directory
    logs_dir_path = os.path.join(out_dir_path, 'lightning_logs/')
    os.makedirs(logs_dir_path, exist_ok=True)
    # Create Lightning checkpoint directory
    ckpt_dir_path = os.path.join(out_dir_path, 'lightning_checkpoints/')
    os.makedirs(ckpt_dir_path, exist_ok=True)
    # Create a temp. directory
    temp_dir_path = os.path.join(out_dir_path, 'temp')
    os.makedirs(temp_dir_path, exist_ok=True)


    # Sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    if hparams.multirun_seed:
        seed_everything(hparams.multirun_seed, workers=True)
    else:
        seed_everything(42, workers=True)

    # Data
    data = CXRDataModule(image_size=IMAGE_SIZE,
                              cxrs_filepath=CXRS_FILEPATH,
                              embeddings_filepath=EMBEDDINGS_FILEPATH,
                              pseudo_rgb=True,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              train_records=TRAIN_RECORDS_CSV,
                              val_records=VAL_RECORDS_CSV,
                              test_records=TEST_RECORDS_CSV)
    
    # Save sample images
    for idx in range(5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir_path, f'sample_{idx}.jpg'), sample['cxr'].astype(np.uint8))

    # Model
    model = CXRModel_LinearProbing.load_from_checkpoint(BEST_CHECKPOINT_FULLPATH, num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE,
                                                        out_dir_path=out_dir_path, target_fpr=TARGET_FPR)


    # WandB logger
    project_name = OUT_DIR_NAME.replace('/', '_').lower().strip('_')
    if hparams.multirun_seed:
        multirun_seed = hparams.multirun_seed
        run_name = f'run_{project_name}_multirun-seed{multirun_seed}_{datetime.now().strftime("%Y%m%d_%H%M")}' 
    else:
        run_name = f'run_{project_name}_{datetime.now().strftime("%Y%m%d_%H%M")}' 
    wandb_logger = WandbLogger(save_dir=logs_dir_path, 
                               project=project_name,
                               name=run_name, 
                               log_model="all")

    # Train
    trainer = Trainer(
        default_root_dir=ckpt_dir_path,
        callbacks=[ModelCheckpoint(monitor='val_loss', 
                                   mode='min', 
                                   filename='best-checkpoint_zsinfer_CXR-model_lp_{epoch}-{val_loss:.4f}',
                                   dirpath=ckpt_dir_path), 
                   TQDMProgressBar(refresh_rate=10),
                   ],
        log_every_n_steps=5,
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=hparams.gpus,
        logger=wandb_logger,
        deterministic=True
    )
    trainer.logger._default_hp_metric = False

    # Final Validating and Testing on the best model just for wandb logs
    model.validation_mode = 'Final'
    trainer.validate(model=model, datamodule=data)
    trainer.test(model=model, datamodule=data)
    save_and_plot_all_metrics(out_dir_path=out_dir_path)

    device = torch.device("cuda:" + str(hparams.dev) if torch.cuda.is_available() else "cpu")
    best_model = model
    best_model.to(device)

    # Generate and Save Outputs
    run_evaluation_phase(model=best_model, dataloader=data.val_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'outputs_val.csv'), phase='validation_outputs', input_type='cxr')
    run_evaluation_phase(model=best_model, dataloader=data.test_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'outputs_test.csv'), phase='testing_outputs', input_type='cxr')
    # Extract and Save Embeddings
    best_model.remove_head()
    run_evaluation_phase(model=best_model, dataloader=data.val_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'embeddings_val.csv'), phase='validation_embeddings', input_type='cxr')
    run_evaluation_phase(model=best_model, dataloader=data.test_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'embeddings_test.csv'), phase='testing_embeddings', input_type='cxr')
    best_model.reset_head()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1, help='Number of GPUs to use')
    parser.add_argument('--dev', default=0, help='GPU device number')
    parser.add_argument('--multirun_seed', default=None, help='Seed for initialising randomness in multiruns for reproducibility')
    parser.add_argument('--inference_on', default='mimic', choices=['chexpert', 'mimic'], help='Dataset module for inference')
    
    args = parser.parse_args()

    main(args)

