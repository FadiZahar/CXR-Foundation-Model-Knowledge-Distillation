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

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar

# Import custom modules
from models.knowledge_distillation.kd_initialisation__CXR_FMKD__MSE import Pre_CXR_FMKD
from data_modules.chexpert_data_module import CheXpertDataModule



image_size = (224, 224)
CXRFM_embeds_size = 1376
num_classes = 14
batch_size = 150
learning_rate = 0.001
epochs = 20
num_workers = 4

cxrs_filepath = '/vol/biodata/data/chest_xray/CheXpert-v1.0/'
embeddings_filepath = '/vol/biomedic3/bglocker/mscproj24/fz221/data/cxrfm_embeddings/chexpert/cxr_numpy'
train_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.train.csv'
val_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.val.csv'
test_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.test.csv' 
main_dir_path = '/vol/biomedic3/bglocker/mscproj24/fz221/outputs/'
out_dir_name = 'CXR-FMKD_full-finetuning/'

# Get base model directory and best checkpoint path
base_model_dir_name = 'CXR-FMKD_KD-initialisation/'
base_model_dir_path = os.path.join(main_dir_path, base_model_dir_name)
base_model_checkpoint_filepath = os.path.join(base_model_dir_path, 'lightning_checkpoints/best-checkpoint_pre-CXR-FMKD.ckpt')

# Ensure the checkpoint file exists
if not os.path.exists(base_model_checkpoint_filepath):
    raise FileNotFoundError(f"CXR-FMKD base model checkpoint file not found: {base_model_checkpoint_filepath}")

# If the file exists, continue
print("CXR-FMKD checkpoint file is available, proceeding with subsequent operations.")



class CXR_FMKD_FullFineTuning(LightningModule):
    def __init__(self, num_classes: int, learning_rate: float, base_lightning_module: LightningModule):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.base_model =  base_lightning_module.model
        self.extract_features = False
         
        # CXR-FMKD: Linear Probing
        num_features = self.base_model.classifier.out_features   # out_features: 1376
        if num_features != CXRFM_embeds_size:
            raise ValueError(f"Expected out_features to be {CXRFM_embeds_size}, but got {num_features}")
        self.classifier = nn.Linear(num_features, self.num_classes)

    def remove_head(self): 
        self.extract_features = True

    def forward(self, x):
        features = self.base_model(x)
        if self.extract_features:
            return features
        else:
            return self.classifier(features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.base_model.parameters()) + list(self.classifier.parameters()), lr=self.learning_rate)
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


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    logits_list = []
    probs_list = []
    targets_list = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader, desc='Evaluate Loop')):
            cxrs, labels = batch['cxr'].to(device), batch['label'].to(device)
            logits = model(cxrs)
            probs = torch.sigmoid(logits)
            logits_list.append(logits)
            probs_list.append(probs)
            targets_list.append(labels)

        logits_array = torch.cat(logits_list, dim=0)
        probs_array = torch.cat(probs_list, dim=0)
        targets_array = torch.cat(targets_list, dim=0)

        counts = []
        for i in range(0, num_classes):
            t = targets_list[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return probs_array.cpu().numpy(), targets_array.cpu().numpy(), logits_array.cpu().numpy()


def extract_embeddings(model, data_loader, device):
    model.eval()
    embeddings_list = []
    targets_list = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader, desc='Extracting Embeddings Loop')):
            cxrs, labels = batch['cxr'].to(device), batch['label'].to(device)
            embeddings = model(cxrs)
            embeddings_list.append(embeddings)
            targets_list.append(labels)

        embeddings_array = torch.cat(embeddings_list, dim=0)
        targets_array = torch.cat(targets_list, dim=0)

    return embeddings_array.cpu().numpy(), targets_array.cpu().numpy()


def run_evaluation_phase(model, dataloader, device, num_classes, file_path, phase):
    print(f'<<>> {phase.upper()} PHASE <<>>')
    if 'embeddings' in phase:
        model.remove_head()
        embeddings, targets = extract_embeddings(model, dataloader, device)
        save_embeddings_to_csv(embeddings, targets, num_classes, file_path)
    else:
        probs, targets, logits = evaluate(model, dataloader, device, num_classes)
        save_predictions_to_csv(probs, logits, targets, num_classes, file_path)


def save_predictions_to_csv(probs, logits, targets, num_classes, file_path):
    cols_names_probs = [f'prob_class_{i}' for i in range(num_classes)]
    cols_names_logits = [f'logit_class_{i}' for i in range(num_classes)]
    cols_names_targets = [f'target_class_{i}' for i in range(num_classes)]
    
    df_probs = pd.DataFrame(data=probs, columns=cols_names_probs)
    df_logits = pd.DataFrame(data=logits, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df = pd.concat([df_probs, df_logits, df_targets], axis=1)
    df.to_csv(file_path, index=False)


def save_embeddings_to_csv(embeddings, targets, num_classes, file_path):
    cols_names_embeddings = [f'embed_{i}' for i in range(embeddings.shape[1])]
    cols_names_targets = [f'target_class_{i}' for i in range(num_classes)]
    
    df_embeddings = pd.DataFrame(data=embeddings, columns=cols_names_embeddings)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df = pd.concat([df_embeddings, df_targets], axis=1)
    df.to_csv(file_path, index=False)


def main(hparams):

    # Sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(42, workers=True)

    # Data
    data = CheXpertDataModule(image_size=image_size,
                              cxrs_filepath=cxrs_filepath,
                              embeddings_filepath=embeddings_filepath,
                              pseudo_rgb=True,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              train_records=train_records_csv,
                              val_records=val_records_csv,
                              test_records=test_records_csv)

    # Model
    model_type = CXR_FMKD_FullFineTuning
    # Load the base student model trained through KD using the teacher CXR-FM embeddings, prior to attaching any classification head
    base_lightning_module = Pre_CXR_FMKD.load_from_checkpoint(base_model_checkpoint_filepath)
    model = model_type(num_classes=num_classes, learning_rate=learning_rate, base_lightning_module=base_lightning_module)

    # Create output directory
    out_dir_path = os.path.join(main_dir_path, out_dir_name)
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
        callbacks=[ModelCheckpoint(monitor='val_loss', mode='min', filename='best-checkpoint_CXR-FMKD_fft_{epoch}-{val_loss:.2f}'), 
                    TQDMProgressBar(refresh_rate=10)],
        log_every_n_steps=5,
        max_epochs=epochs,
        accelerator='auto',
        devices=hparams.gpus,
        logger=TensorBoardLogger(logs_dir_path, name=out_dir_name.lower()),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model=model, datamodule=data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    device = torch.device("cuda:" + str(hparams.dev) if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Get Outputs
    run_evaluation_phase(model, data.val_dataloader(), device, os.path.join(out_dir_path, 'outputs.val.csv'), 'validation_outputs')
    run_evaluation_phase(model, data.test_dataloader(), device, os.path.join(out_dir_path, 'outputs.test.csv'), 'testing_outputs')
    # Extract Embeddings
    run_evaluation_phase(model, data.val_dataloader(), device, os.path.join(out_dir_path, 'embeddings.val.csv'), 'validation_embeddings')
    run_evaluation_phase(model, data.test_dataloader(), device, os.path.join(out_dir_path, 'embeddings.test.csv'), 'testing_embeddings')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    main(args)

