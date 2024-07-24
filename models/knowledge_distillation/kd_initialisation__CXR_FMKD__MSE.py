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



image_size = (224, 224)
CXRFM_embeds_size = 1376
batch_size = 150
learning_rate = 0.001
epochs = 20
num_workers = 4

cxrs_filepath = '/vol/biodata/data/chest_xray/CheXpert-v1.0/'
embeddings_filepath = '/vol/biomedic3/bglocker/mscproj24/fz221/data/cxrfm_embeddings/chexpert/cxr_numpy'
train_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.train.csv'
val_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.val.csv'
# test_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.test.csv' --> should be reserved for downstream task fine-tuning
main_dir_path = '/vol/biomedic3/bglocker/mscproj24/fz221/outputs/'
out_dir_name = 'CXR-FMKD_KD-initialisation-MSE/'



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


def evaluate(model, data_loader, device):
    model.eval()
    output_embeds_list = []
    target_embeds_list = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader, desc='Evaluate Loop')):
            cxrs, target_embeds = batch['cxr'].to(device), batch['embedding'].to(device)
            output_embeds = model(cxrs)
            output_embeds_list.append(output_embeds)
            target_embeds_list.append(target_embeds)

        output_embeds_array = torch.cat(output_embeds_list, dim=0)
        target_embeds_array = torch.cat(target_embeds_list, dim=0)

    return output_embeds_array.cpu().numpy(), target_embeds_array.cpu().numpy()


def run_evaluation_phase(model, dataloader, device, file_path, phase):
    print(f'<<>> {phase.upper()} PHASE <<>>')
    if 'embeddings' in phase:
        model.remove_head()
        pre_embeds, target_embeds = evaluate(model, dataloader, device)
        save_embeddings_to_csv(pre_embeds, target_embeds, file_path)
    else:
        output_embeds, target_embeds = evaluate(model, dataloader, device)
        save_embeddings_to_csv(output_embeds, target_embeds, file_path)


def save_embeddings_to_csv(embeds, target_embeds, file_path):
    cols_names_embeds = [f'embed_{i}' for i in range(embeds.shape[1])]
    cols_names_target_embeds = [f'target_embed_{i}' for i in range(target_embeds.shape[1])]
    
    df_embeddings = pd.DataFrame(data=embeds, columns=cols_names_embeds)
    df_targets = pd.DataFrame(data=target_embeds, columns=cols_names_target_embeds)
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
                              dev_split=[0.7, 0.3])

    # Model
    model_type = Pre_CXR_FMKD
    model = model_type(learning_rate=learning_rate, embedding_size=CXRFM_embeds_size)

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
        callbacks=[ModelCheckpoint(monitor='val_loss', 
                                   mode='min', 
                                   filename='best-checkpoint_pre-CXR-FMKD',
                                   dirpath=ckpt_dir_path), 
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

