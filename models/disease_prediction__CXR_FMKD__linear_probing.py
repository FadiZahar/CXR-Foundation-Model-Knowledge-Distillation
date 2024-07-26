import os
import numpy as np
from skimage.io import imsave
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar


# Import custom modules
from models.knowledge_distillation.kd_initialisation__CXR_FMKD__MSE import Pre_CXR_FMKD
from data_modules.chexpert_data_module import CheXpertDataModule
from utils.output_utils.generate_and_save_outputs import run_evaluation_phase

# Import global variables
from config.config_chexpert import IMAGE_SIZE, CXRFM_EMBEDS_SIZE, NUM_CLASSES, EPOCHS, NUM_WORKERS, BATCH_SIZE, LEARNING_RATE
from config.config_chexpert import CXRS_FILEPATH, EMBEDDINGS_FILEPATH, TRAIN_RECORDS_CSV, VAL_RECORDS_CSV, TEST_RECORDS_CSV, MAIN_DIR_PATH

OUT_DIR_NAME = 'CXR-FMKD_linear-probing/'


# Get base model directory and best checkpoint path
BASE_MODEL_DIR_NAME = 'CXR-FMKD_KD-initialisation/'
BASE_MODEL_DIR_PATH = os.path.join(MAIN_DIR_PATH, BASE_MODEL_DIR_NAME)
BASE_MODEL_CHECKPOINT_FILEPATH = os.path.join(BASE_MODEL_DIR_PATH, 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD.ckpt')

# Ensure the checkpoint file exists
if not os.path.exists(BASE_MODEL_CHECKPOINT_FILEPATH):
    raise FileNotFoundError(f"CXR-FMKD base model checkpoint file not found: {BASE_MODEL_CHECKPOINT_FILEPATH}")

# If the file exists, continue
print("CXR-FMKD checkpoint file is available, proceeding with subsequent operations.")



class CXR_FMKD_LinearProbing(LightningModule):
    def __init__(self, num_classes: int, learning_rate: float, embedding_size: int, base_lightning_module: LightningModule):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        self.base_model =  base_lightning_module.model
        self.extract_features = False

        # Ensure the base model is frozen for linear probing
        freeze_model(self.base_model)
         
        # CXR-FMKD: Linear Probing
        num_features = self.base_model.classifier.out_features   # out_features: 1376
        if num_features != self.embedding_size:
            raise ValueError(f"Expected out_features to be {self.embedding_size}, but got {num_features}")
        self.classifier = nn.Linear(num_features, self.num_classes)

    def remove_head(self): 
        self.extract_features = True

    def reset_head(self): 
        self.extract_features = False

    def forward(self, x):
        features = self.base_model(x)
        if self.extract_features:
            return features
        else:
            return self.classifier(features)

    def configure_optimizers(self):
        params_to_update = self.classifier.parameters()
        optimizer = torch.optim.Adam(params_to_update, lr=self.learning_rate)
        return optimizer

    def unpack_batch(self, batch):
        return batch['cxr'], batch['label']

    def process_batch(self, batch):
        cxrs, labels = self.unpack_batch(batch)   # cxrs: Chest X-Rays, embeds: embeddings
        logits = self.forward(cxrs)
        probs = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(probs, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, prog_bar=True)
        grid = torchvision.utils.make_grid(batch['cxr'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('Chest X-Rays', grid, self.global_step)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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
    model_type = CXR_FMKD_LinearProbing
    # Load the base student model trained through KD using the teacher CXR-FM embeddings, prior to attaching any classification head
    base_lightning_module = Pre_CXR_FMKD.load_from_checkpoint(BASE_MODEL_CHECKPOINT_FILEPATH)
    model = model_type(num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, embedding_size=CXRFM_EMBEDS_SIZE, base_lightning_module=base_lightning_module)

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

    # Train
    trainer = Trainer(
        default_root_dir=ckpt_dir_path,
        callbacks=[ModelCheckpoint(monitor='val_loss', 
                                   mode='min', 
                                   filename='best-checkpoint_CXR-FMKD_lp_{epoch}-{val_loss:.2f}',
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
    run_evaluation_phase(model=model, dataloader=data.val_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'outputs_val.csv'), phase='validation_outputs', input_type='cxr')
    run_evaluation_phase(model=model, dataloader=data.test_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'outputs_test.csv'), phase='testing_outputs', input_type='cxr')
    # Extract and Save Embeddings
    model.remove_head()
    run_evaluation_phase(model=model, dataloader=data.val_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'embeddings_val.csv'), phase='validation_embeddings', input_type='cxr')
    run_evaluation_phase(model=model, dataloader=data.test_dataloader(), device=device, num_classes=NUM_CLASSES, 
                         file_path=os.path.join(out_dir_path, 'embeddings_test.csv'), phase='testing_embeddings', input_type='cxr')
    model.reset_head()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    main(args)

