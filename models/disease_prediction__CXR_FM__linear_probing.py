import os
import numpy as np
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


# Import custom modules
from data_modules.chexpert_data_module import CheXpertDataModule
from utils.output_utils.generate_and_save_raw_outputs import run_evaluation_phase
from utils.output_utils.generate_and_save_metrics import generate_and_log_metrics, save_and_plot_all_metrics
from utils.callback_utils.training_callbacks import TrainLoggingCallback

# Import global variables
from config.config_chexpert import IMAGE_SIZE, CXRFM_EMBEDS_SIZE, NUM_CLASSES, EPOCHS, NUM_WORKERS, BATCH_SIZE, LEARNING_RATE, TARGET_FPR
from config.config_chexpert import CXRS_FILEPATH, EMBEDDINGS_FILEPATH, TRAIN_RECORDS_CSV, VAL_RECORDS_CSV, TEST_RECORDS_CSV, MAIN_DIR_PATH

OUT_DIR_NAME = 'CXR-FM_linear-probing/'



class CXR_FM(LightningModule):
    def __init__(self, num_classes: int, learning_rate: float, embedding_size: int, out_dir_path:str, target_fpr: float):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.out_dir_path = out_dir_path
        self.target_fpr = target_fpr
        self.validation_step_outputs = []
        self.testing_step_outputs = []
        self.validation_mode = 'Training'
        self.extract_features = False

        # log hyperparameters
        self.save_hyperparameters()
        
        # CXR-FM: linear probing
        self.model = nn.Sequential(
            nn.Linear(self.embedding_size, self.num_classes)
        )

    def remove_head(self): 
        self.extract_features = True

    def reset_head(self):
        self.extract_features = False

    def forward(self, x):
        if self.extract_features:
            return x
        else:
            return self.model(x)

    def configure_optimizers(self):
        params_to_update = [param for param in self.parameters() if param.requires_grad]
        optimizer = torch.optim.Adam(params_to_update, lr=self.learning_rate)
        return optimizer

    def unpack_batch(self, batch):
        return batch['embedding'], batch['label']

    def process_batch(self, batch):
        embeddings, labels = self.unpack_batch(batch)
        logits = self.forward(embeddings)
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
    model_type = CXR_FM
    model = model_type(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, embedding_size=CXRFM_EMBEDS_SIZE, 
                       out_dir_path=out_dir_path, target_fpr=TARGET_FPR)
    
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
                                   filename='best-checkpoint_CXR-FM_lp_{epoch}-{val_loss:.4f}',
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
                         file_path=os.path.join(out_dir_path, 'outputs_val.csv'), phase='validation_outputs', input_type='embedding')
    run_evaluation_phase(model=best_model, dataloader=data.test_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'outputs_test.csv'), phase='testing_outputs', input_type='embedding')
    # Extract and Save Embeddings
    best_model.remove_head()
    run_evaluation_phase(model=best_model, dataloader=data.val_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'embeddings_val.csv'), phase='validation_embeddings', input_type='embedding')
    run_evaluation_phase(model=best_model, dataloader=data.test_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'embeddings_test.csv'), phase='testing_embeddings', input_type='embedding')
    best_model.reset_head()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1, help='Number of GPUs to use')
    parser.add_argument('--dev', default=0, help='GPU device number')
    args = parser.parse_args()

    main(args)

