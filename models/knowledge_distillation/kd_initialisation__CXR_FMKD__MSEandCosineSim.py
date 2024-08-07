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
from utils.output_utils.kd_generate_and_save_raw_outputs import run_evaluation_phase
from utils.callback_utils.training_callbacks import TrainLoggingCallback

# Import global variables
from config.config_chexpert import IMAGE_SIZE, CXRFM_EMBEDS_SIZE, NUM_WORKERS, BATCH_SIZE, LEARNING_RATE
from config.config_chexpert import CXRS_FILEPATH, EMBEDDINGS_FILEPATH, TRAIN_RECORDS_CSV, VAL_RECORDS_CSV, MAIN_DIR_PATH

DEV_SPLIT = [0.7, 0.3]
EPOCHS = 40
OUT_DIR_NAME = 'CXR-FMKD_KD-initialisation-MSEandCosineSim/'



class Pre_CXR_FMKD(LightningModule):
    def __init__(self, learning_rate: float, embedding_size: int):
        super().__init__()
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.validation_mode = 'Training'

        # log hyperparameters
        self.save_hyperparameters()
        
        # KD from teacher (CXR-FM) to student (DenseNet-169)
        self.model = models.densenet169(weights=models.DenseNet169_Weights.DEFAULT)
        self.num_features = self.model.classifier.in_features   # in_features: 1664 | out_features: 1000 (ImageNet)

        # Replace original classifier with new f.c. layer mapping the 1664 input features to 1376 (to match CXR-FM's embeddings), and store it:
        self.classifier = nn.Linear(self.num_features, self.embedding_size)
        self.model.classifier = self.classifier  

    def remove_head(self): 
        self.model.classifier = nn.Identity(self.num_features)
    
    def reset_head(self):
        self.model.classifier = self.classifier

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params_to_update = [param for param in self.parameters() if param.requires_grad]
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
        return batch['cxr'], batch['embedding']

    def process_batch(self, batch):
        cxrs, target_embeds = self.unpack_batch(batch)   # cxrs: Chest X-Rays, embeds: embeddings
        output_embeds = self.forward(cxrs)
        # MSE Loss
        mse_loss = F.mse_loss(output_embeds, target_embeds)
        # Cosine Similarity Loss
        cosine_sim = F.cosine_similarity(output_embeds, target_embeds, dim=1)
        cosine_loss = 1 - cosine_sim.mean()
        # Combine losses
        alpha = 0.5  # weight for MSE loss
        beta = 0.5   # weight for cosine similarity loss
        loss = alpha * mse_loss + beta * cosine_loss
        return loss

    ## Training
    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Log the current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False)

        if batch_idx == 0 and batch['cxr'].shape[0] >= 20:
            grid = torchvision.utils.make_grid(batch['cxr'][0:20, ...], nrow=5, normalize=True)
            grid = grid.permute(1, 2, 0).cpu().numpy()
            wandb.log({"Chest X-Rays": [wandb.Image(grid, caption=f"Batch {batch_idx}")]}, step=self.global_step)

        return loss

    ## Validation
    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        if self.validation_mode == 'Training':
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log('val_final_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss}

    ## Testing
    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'test_loss': loss}


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False



def main(hparams):

    # Create output directory
    if hparams.multirun_id:
        inner_out_dir_name = f"{OUT_DIR_NAME.strip('/')}_{hparams.multirun_id}"
        out_dir_path = os.path.join(MAIN_DIR_PATH, OUT_DIR_NAME, 'multiruns', inner_out_dir_name)
    else:
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
    
    # Save sample images
    for idx in range(5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir_path, f'sample_{idx}.jpg'), sample['cxr'].astype(np.uint8))

    # Model
    model_type = Pre_CXR_FMKD
    model = model_type(learning_rate=LEARNING_RATE, embedding_size=CXRFM_EMBEDS_SIZE)

    # Callback metric logging
    train_logger = TrainLoggingCallback(filename=os.path.join(logs_dir_path, 'val_loss_step.csv'))

    # WandB logger
    project_name = OUT_DIR_NAME.replace('/', '_').lower().strip('_')
    if hparams.multirun_id:
        multirun_id = hparams.multirun_id
        run_name = f'run_{project_name}_{multirun_id}_{datetime.now().strftime("%Y%m%d_%H%M")}' 
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
                                   filename='best-checkpoint_pre-CXR-FMKD_MSEandCosineSim_{epoch}-{val_loss:.4f}',
                                   dirpath=ckpt_dir_path), 
                   TQDMProgressBar(refresh_rate=10),
                   train_logger],
        log_every_n_steps=5,
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=hparams.gpus,
        logger=wandb_logger,
        deterministic=True
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model=model, datamodule=data)

    # Final Validating and Testing on the best model just for wandb logs
    model.validation_mode = 'Final'
    trainer.validate(model=model, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path)
    trainer.test(model=model, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path)

    best_model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    device = torch.device("cuda:" + str(hparams.dev) if torch.cuda.is_available() else "cpu")
    best_model.to(device)

    # Generate and Save Outputs
    run_evaluation_phase(model=best_model, dataloader=data.val_dataloader(), device=device, file_path=os.path.join(out_dir_path, 'outputs_val.csv'), 
                         phase='validation_outputs')
    run_evaluation_phase(model=best_model, dataloader=data.test_dataloader(), device=device, file_path=os.path.join(out_dir_path, 'outputs_test.csv'), 
                         phase='testing_outputs')
    # Extract and Save Embeddings
    best_model.remove_head()
    run_evaluation_phase(model=best_model, dataloader=data.val_dataloader(), device=device, file_path=os.path.join(out_dir_path, 'embeddings_val.csv'), 
                         phase='validation_embeddings')
    run_evaluation_phase(model=best_model, dataloader=data.test_dataloader(), device=device, file_path=os.path.join(out_dir_path, 'embeddings_test.csv'), 
                         phase='testing_embeddings')
    best_model.reset_head()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1, help='Number of GPUs to use')
    parser.add_argument('--dev', default=0, help='GPU device number')
    parser.add_argument('--multirun_id', default=None, help='Optional identifier for multi runs')
    
    args = parser.parse_args()

    main(args)

