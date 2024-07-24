import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imsave
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


# Import custom modules
from data_modules.chexpert_data_module import CheXpertDataModule
from utils.output_utils.kd_generate_and_save_outputs import run_evaluation_phase

# Import global variables
from config.config_chexpert import IMAGE_SIZE, CXRFM_EMBEDS_SIZE, EPOCHS, NUM_WORKERS, BATCH_SIZE, LEARNING_RATE
from config.config_chexpert import CXRS_FILEPATH, EMBEDDINGS_FILEPATH, TRAIN_RECORDS_CSV, VAL_RECORDS_CSV, MAIN_DIR_PATH

DEV_SPLIT = [0.7, 0.3]
OUT_DIR_NAME = 'CXR-FMKD_KD-initialisation-MSE/'



class Pre_CXR_FMKD(LightningModule):
    def __init__(self, learning_rate: float, embedding_size: int):
        super().__init__()
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.save_hyperparameters()
        
        # KD from teacher (CXR-FM) to student (DenseNet-169)
        self.model = models.densenet169(pretrained=True)
        num_features = self.model.classifier.in_features   # in_features: 1664 | out_features: 1000 (ImageNet)
        # Replace original classifier with new f.c. layer mapping the 1664 input features to 1376 (to match CXR-FM's embeddings):
        self.model.classifier = nn.Linear(num_features, embedding_size)  

    def remove_head(self): 
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity(num_features)

    def forward(self, x):
        return self.model.forward(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=self.learning_rate)
        return optimizer

    def unpack_batch(self, batch):
        return batch['cxr'], batch['embedding']

    def process_batch(self, batch):
        cxrs, target_embeds = self.unpack_batch(batch)   # cxrs: Chest X-Rays, embeds: embeddings
        output_embeds = self.forward(cxrs)
        # Calculate Mean Squared Error (MSE) Loss between output embeddings from DenseNet-169 and target embeddings from CXR-FM
        loss = F.mse_loss(output_embeds, target_embeds)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, prog_bar=True)
        if batch_idx == 0:
            grid = torchvision.utils.make_grid(batch['cxr'][0:4, ...], nrow=2, normalize=True)
            self.logger.experiment.add_image('Chest X-Rays', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def main(hparams):

    # Sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(42, workers=True)

    # Data
    data = CheXpertDataModule(image_size=IMAGE_SIZE,
                              cxrs_filepath=CXRS_FILEPATH,
                              embeddings_filepath=EMBEDDINGS_FILEPATH,
                              pseudo_rgb=True,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              train_records=TRAIN_RECORDS_CSV,
                              val_records=VAL_RECORDS_CSV,
                              dev_split=DEV_SPLIT)

    # Model
    model_type = Pre_CXR_FMKD
    model = model_type(learning_rate=LEARNING_RATE, embedding_size=CXRFM_EMBEDS_SIZE)

    # Create output directory
    out_dir_path = os.path.join(MAIN_DIR_PATH, OUT_DIR_NAME)
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
    for idx in range(5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir_path, 'sample_' + str(idx) + '.jpg'), sample['cxr'].astype(np.uint8))

    # Train
    trainer = Trainer(
        default_root_dir=ckpt_dir_path,
        callbacks=[ModelCheckpoint(monitor='val_loss', 
                                   mode='min', 
                                   filename='best-checkpoint_pre-CXR-FMKD',
                                   dirpath=ckpt_dir_path), 
                    TQDMProgressBar(refresh_rate=10)],
        log_every_n_steps=5,
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=hparams.gpus,
        logger=TensorBoardLogger(logs_dir_path, name=OUT_DIR_NAME.lower()),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model=model, datamodule=data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    device = torch.device("cuda:" + str(hparams.dev) if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate and Save Outputs
    run_evaluation_phase(model=model, dataloader=data.val_dataloader(), device=device, file_path=os.path.join(out_dir_path, 'outputs_val.csv'), 
                         phase='validation_outputs')
    run_evaluation_phase(model=model, dataloader=data.test_dataloader(), device=device, file_path=os.path.join(out_dir_path, 'outputs_test.csv'), 
                         phase='testing_outputs')
    # Extract and Save Embeddings
    run_evaluation_phase(model=model, dataloader=data.val_dataloader(), device=device, file_path=os.path.join(out_dir_path, 'embeddings_val.csv'), 
                         phase='validation_embeddings')
    run_evaluation_phase(model=model, dataloader=data.test_dataloader(), device=device, file_path=os.path.join(out_dir_path, 'embeddings_test.csv'), 
                         phase='testing_embeddings')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    main(args)

