import os
import numpy as np
from skimage.io import imsave
from argparse import ArgumentParser
from datetime import datetime
import wandb

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
from utils.output_utils.generate_and_save_outputs import run_evaluation_phase
from utils.callback_utils.training_callbacks import MetricLoggingCallback

# Import global variables
from config.config_chexpert import IMAGE_SIZE, NUM_CLASSES, EPOCHS, NUM_WORKERS, BATCH_SIZE, LEARNING_RATE
from config.config_chexpert import CXRS_FILEPATH, EMBEDDINGS_FILEPATH, TRAIN_RECORDS_CSV, VAL_RECORDS_CSV, TEST_RECORDS_CSV, MAIN_DIR_PATH

OUT_DIR_NAME = 'CheXpert-model_full-finetuning/'



class DenseNet(LightningModule):
    def __init__(self, num_classes: int, learning_rate: float):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # log hyperparameters
        self.save_hyperparameters()
        
        # DenseNet-169: full finetuning
        self.model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        num_features = self.model.classifier.in_features   # in_features: 1664 | out_features: 1000 (ImageNet)
        # Replace original classifier with new f.c. layer mapping the 1664 input features to 14 (disease classes):
        self.model.classifier = nn.Linear(num_features, self.num_classes)  

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
        base_lr = self.learning_rate
        max_lr = self.learning_rate*10
        optimizer = torch.optim.Adam(params_to_update, lr=base_lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=self.trainer.estimated_stepping_batches  
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }

    def unpack_batch(self, batch):
        return batch['cxr'], batch['label']

    def process_batch(self, batch):
        cxrs, labels = self.unpack_batch(batch)   # cxrs: Chest X-Rays
        logits = self.forward(cxrs)
        probs = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(probs, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, prog_bar=True)
        grid = torchvision.utils.make_grid(batch['cxr'][0:4, ...], nrow=2, normalize=True)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        wandb.log({"Chest X-Rays": [wandb.Image(grid, caption=f"Batch {batch_idx}")]}, step=self.global_step, commit=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {"val_loss": loss}

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
                              test_records=TEST_RECORDS_CSV)

    # Model
    model_type = DenseNet
    model = model_type(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE)

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

    # Save sample images
    for idx in range(5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir_path, 'sample_' + str(idx) + '.jpg'), sample['cxr'].astype(np.uint8))

    # Callback metric logging
    metric_logger = MetricLoggingCallback(filename=os.path.join(logs_dir_path, 'metrics.csv'))

    # WandB logger
    project_name = OUT_DIR_NAME.replace('/', '_').lower().strip('_')
    wandb_logger = WandbLogger(save_dir=logs_dir_path, 
                               project=project_name,
                               name='run_' + project_name + '_' + datetime.now().strftime('%Y%m%d_%H%M'), 
                               log_model="all")

    # Train
    trainer = Trainer(
        default_root_dir=ckpt_dir_path,
        callbacks=[ModelCheckpoint(monitor='val_loss', 
                                   mode='min', 
                                   filename='best-checkpoint_CheXpert-model_fft_{epoch}-{val_loss:.2f}',
                                   dirpath=ckpt_dir_path), 
                   TQDMProgressBar(refresh_rate=10),
                   metric_logger],
        log_every_n_steps=5,
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=hparams.gpus,
        logger=wandb_logger,
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model=model, datamodule=data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=NUM_CLASSES)
    device = torch.device("cuda:" + str(hparams.dev) if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Generate and Save Outputs
    run_evaluation_phase(model=model, dataloader=data.val_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'outputs_val.csv'), phase='validation_outputs', input_type='cxr')
    run_evaluation_phase(model=model, dataloader=data.test_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'outputs_test.csv'), phase='testing_outputs', input_type='cxr')
    # Extract and Save Embeddings
    run_evaluation_phase(model=model, dataloader=data.val_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'embeddings_val.csv'), phase='validation_embeddings', input_type='cxr')
    run_evaluation_phase(model=model, dataloader=data.test_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'embeddings_test.csv'), phase='testing_embeddings', input_type='cxr')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    main(args)

