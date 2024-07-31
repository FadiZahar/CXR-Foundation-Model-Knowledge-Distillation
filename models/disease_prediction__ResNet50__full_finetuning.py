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
from utils.output_utils.generate_and_save_raw_outputs import run_evaluation_phase
from utils.output_utils.generate_and_save_metrics import generate_and_log_metrics, save_and_plot_all_metrics
from utils.callback_utils.training_callbacks import TrainLoggingCallback

# Import global variables
from config.config_chexpert import IMAGE_SIZE, NUM_CLASSES, EPOCHS, NUM_WORKERS, BATCH_SIZE, LEARNING_RATE, TARGET_FPR
from config.config_chexpert import CXRS_FILEPATH, EMBEDDINGS_FILEPATH, TRAIN_RECORDS_CSV, VAL_RECORDS_CSV, TEST_RECORDS_CSV, MAIN_DIR_PATH

OUT_DIR_NAME = 'ResNet50_full-finetuning/'



class ResNet50(LightningModule):
    def __init__(self, num_classes: int, learning_rate: float, out_dir_path:str, target_fpr: float):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.out_dir_path = out_dir_path
        self.target_fpr = target_fpr
        self.validation_step_outputs = []
        self.testing_step_outputs = []
        self.validation_mode = 'Training'

        # log hyperparameters
        self.save_hyperparameters()
        
        # ResNet-50: full finetuning
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.num_features = self.model.fc.in_features   # in_features: 2048 | out_features: 1000 (ImageNet)

        # Replace original f.c. layer with new f.c. layer mapping the 2048 input features to 14 (disease classes), and store it:
        self.fc = nn.Linear(self.num_features, self.num_classes)
        self.model.fc = self.fc 

    def remove_head(self): 
        self.model.fc = nn.Identity(self.num_features)
    
    def reset_head(self):
        self.model.fc = self.fc

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
        return batch['cxr'], batch['label']

    def process_batch(self, batch):
        cxrs, labels = self.unpack_batch(batch)   # cxrs: Chest X-Rays
        logits = self.forward(cxrs)
        probs = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(probs, labels)
        return logits, probs, labels, loss


    ## Training
    def training_step(self, batch, batch_idx):
        _, _, _, loss = self.process_batch(batch)
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
        logits, probs, labels, loss = self.process_batch(batch)
        if self.validation_mode == 'Training':
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        else:
            self.log('val_final_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Convert tensors to CPU and numpy for sklearn compatibility
        labels_np = labels.cpu().numpy()
        probs_np = probs.cpu().detach().numpy()

        output = {'val_loss': loss, 'logits': logits, 'probs': probs_np, 'labels': labels_np}
        self.validation_step_outputs.append(output)
        return output
    
    def on_validation_epoch_end(self):
        all_probs = np.vstack([x['probs'] for x in self.validation_step_outputs])
        all_labels = np.vstack([x['labels'] for x in self.validation_step_outputs])
        # Check the mode and log accordingly
        if self.validation_mode == 'Training':
            generate_and_log_metrics(targets=all_labels, probs=all_probs, out_dir_path=self.out_dir_path, 
                                     phase='Validation - During Training', target_fpr=self.target_fpr)
        else:
            generate_and_log_metrics(targets=all_labels, probs=all_probs, out_dir_path=self.out_dir_path, 
                                     phase='Validation - Final', target_fpr=self.target_fpr)
        self.validation_step_outputs.clear()


    ## Testing
    def test_step(self, batch, batch_idx):
        logits, probs, labels, loss = self.process_batch(batch)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # Convert tensors to CPU and numpy for sklearn compatibility
        labels_np = labels.cpu().numpy()
        probs_np = probs.cpu().detach().numpy()

        output = {'test_loss': loss, 'logits': logits, 'probs': probs_np, 'labels': labels_np}
        self.testing_step_outputs.append(output)
        return output
    
    def on_test_epoch_end(self):
        all_probs = np.vstack([x['probs'] for x in self.testing_step_outputs])
        all_labels = np.vstack([x['labels'] for x in self.testing_step_outputs])
        generate_and_log_metrics(targets=all_labels, probs=all_probs, out_dir_path=self.out_dir_path, 
                                 phase='Testing', target_fpr=self.target_fpr)
        self.testing_step_outputs.clear()


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False



def main(hparams):

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


    # Sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(42, workers=True)

    # Data
    data = CheXpertDataModule(image_size=IMAGE_SIZE,
                              cxrs_filepath=CXRS_FILEPATH,
                              embeddings_filepath=EMBEDDINGS_FILEPATH,
                              pseudo_rgb=True,
                              batch_size=128,
                              num_workers=NUM_WORKERS,
                              train_records=TRAIN_RECORDS_CSV,
                              val_records=VAL_RECORDS_CSV,
                              test_records=TEST_RECORDS_CSV)
    
    # Save sample images
    for idx in range(5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir_path, f'sample_{idx}.jpg'), sample['cxr'].astype(np.uint8))

    # Model
    model_type = ResNet50
    model = model_type(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, out_dir_path=out_dir_path, target_fpr=TARGET_FPR)

    # Callback metric logging
    train_logger = TrainLoggingCallback(filename=os.path.join(logs_dir_path, 'val_loss_step.csv'))

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
                                   filename='best-checkpoint_ResNet50_fft_{epoch}-{val_loss:.4f}',
                                   dirpath=ckpt_dir_path), 
                   TQDMProgressBar(refresh_rate=10),
                   train_logger],
        log_every_n_steps=5,
        max_epochs=EPOCHS,
        accelerator='auto',
        devices=hparams.gpus,
        logger=wandb_logger,
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model=model, datamodule=data)

    # Final Validating and Testing on the best model just for wandb logs
    model.validation_mode = 'Final'
    trainer.validate(model=model, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path)
    trainer.test(model=model, datamodule=data, ckpt_path=trainer.checkpoint_callback.best_model_path)
    save_and_plot_all_metrics(out_dir_path=out_dir_path)

    best_model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=NUM_CLASSES)
    device = torch.device("cuda:" + str(hparams.dev) if torch.cuda.is_available() else "cpu")
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
    args = parser.parse_args()

    main(args)

