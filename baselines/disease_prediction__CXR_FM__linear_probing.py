import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread
from skimage.io import imsave
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from torchvision import models

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar



CXRFM_embeds_size = 1376
num_classes = 14
batch_size = 150
learning_rate = 0.001
epochs = 20
num_workers = 4

data_filepath = '/vol/biomedic3/bglocker/cxr-foundation/outputs/chexpert/cxr_numpy/'
train_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.train.csv'
val_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.val.csv'
test_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.test.csv'
main_dir_path = '/vol/biomedic3/bglocker/mscproj24/fz221/outputs/'
out_dir_name = 'CXR-FM_linear-probing/'



class CheXpertDataset(Dataset):
    def __init__(self, records_filepath: str, data_filepath: str):
        self.entries = pd.read_csv(records_filepath)
        self.data_filepath = data_filepath

        self.labels = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

        self.records = []
        for _, row in tqdm(self.entries.iterrows(), total=len(self.entries), desc='Loading CXR Records'):
            image_filepath = row['path_preproc']   # e.g., 'preproc_224x224/patient24428_study22_view1_frontal.jpg'
            # Labels are set to 1 for positive findings and 0 for all other cases (negative, uncertain, or unmentioned).
            label = np.array([row[label.strip()] == 1 for label in self.labels], dtype='float32')
            record = {'image_filepath': image_filepath, 'label': label}
            self.records.append(record)

    def get_sample(self, item):
        record = self.records[item]
        patient_filename = os.path.basename(record['image_filepath'])   # e.g., 'patient24428_study22_view1_frontal.jpg'
        embedding_filepath = os.path.join(self.data_filepath, patient_filename.replace('.jpg', '.dat'))   # e.g., '<data_filepath>/patient24428_study22_view1_frontal.dat'
        embedding = np.fromfile(embedding_filepath, dtype=np.float32)
        return {'embedding': embedding, 'label': record['label']}

    def __getitem__(self, item):
        sample = self.get_sample(item)
        embedding_tensor = torch.from_numpy(sample['embedding'])
        label_tensor = torch.from_numpy(sample['label'])
        return {'embedding': embedding_tensor, 'label': label_tensor}

    def __len__(self):
        return len(self.entries)


class CheXpertDataModule(LightningDataModule):
    def __init__(self, train_records: str, val_records: str, test_records: str, data_filepath: str, batch_size: int, num_workers: int):
        super().__init__()
        self.train_records = train_records
        self.val_records = val_records
        self.test_records = test_records
        self.data_filepath = data_filepath
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = CheXpertDataset(records_filepath=self.train_records, data_filepath=self.data_filepath)
        self.val_set = CheXpertDataset(records_filepath=self.val_records, data_filepath=self.data_filepath)
        self.test_set = CheXpertDataset(records_filepath=self.test_records, data_filepath=self.data_filepath)

        print('>> train_set size: ', len(self.train_set))
        print('>> val_set size:   ', len(self.val_set))
        print('>> test_set size:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class CXR_FM(LightningModule):
    def __init__(self, num_classes: int, learning_rate: float, embedding_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        
        # CXR-FM: linear probing
        self.model = nn.Sequential(
            nn.Linear(embedding_size, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=self.learning_rate)
        return optimizer

    def unpack_batch(self, batch):
        return batch['embedding'], batch['label']

    def process_batch(self, batch):
        embeddings, labels = self.unpack_batch(batch)
        logits = self.forward(embeddings)
        probs = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(probs, labels)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    logits_list = []
    probs_list = []
    targets_list = []

    with torch.no_grad():
        for _, batch in enumerate(tqdm(data_loader, desc='Evaluate-loop')):
            embeddings, labels = batch['embedding'].to(device), batch['label'].to(device)
            logits = model(embeddings)
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


def run_evaluation_phase(model, dataloader, device, num_classes, file_path, phase):
    print(f'<<>> {phase.upper()} PHASE <<>>')
    probs, targets, logits = evaluate(model, dataloader, device, num_classes)
    save_predictions_to_csv(probs, logits, targets, file_path)


def save_predictions_to_csv(probs, logits, targets, file_path):
    cols_names_probs = [f'prob_class_{i}' for i in range(num_classes)]
    cols_names_logits = [f'logit_class_{i}' for i in range(num_classes)]
    cols_names_targets = [f'target_class_{i}' for i in range(num_classes)]
    
    df_probs = pd.DataFrame(data=probs, columns=cols_names_probs)
    df_logits = pd.DataFrame(data=logits, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets, columns=cols_names_targets)
    df = pd.concat([df_probs, df_logits, df_targets], axis=1)
    df.to_csv(file_path, index=False)


def main(hparams):

    # Sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(42, workers=True)

    # Data
    data = CheXpertDataModule(train_records=train_records_csv,
                              val_records=val_records_csv,
                              test_records=test_records_csv,
                              data_filepath=data_filepath,
                              batch_size=batch_size,
                              num_workers=num_workers)

    # Model
    model_type = CXR_FM
    model = model_type(num_classes=num_classes, learning_rate=learning_rate, embedding_size=CXRFM_embeds_size)

    # Create output directory
    out_dir_path = os.path.join(main_dir_path, out_dir_name)
    os.makedirs(out_dir_path, exist_ok=True)
    # Create TensorBoard logs directory
    logs_dir_path = os.path.join(out_dir_path, 'lightning_logs/')
    os.makedirs(logs_dir_path, exist_ok=True)

    # Train
    trainer = Trainer(
        default_root_dir=out_dir_path,
        callbacks=[ModelCheckpoint(monitor='val_loss', mode='min', filename='best-checkpoint_CXR_FM_lp_{epoch}-{val_loss:.2f}'), 
                   TQDMProgressBar(refresh_rate=10)],
        log_every_n_steps=5,
        max_epochs=epochs,
        accelerator='auto',
        devices=hparams.gpus,
        logger=TensorBoardLogger(logs_dir_path, name=out_dir_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model=model, datamodule=data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=num_classes)
    device = torch.device("cuda:" + str(hparams.dev) if torch.cuda.is_available() else "cpu")
    model.to(device)

    run_evaluation_phase(model, data.val_dataloader(), device, num_classes, os.path.join(out_dir_path, 'outputs.val.csv'), 'validation_outputs')
    run_evaluation_phase(model, data.test_dataloader(), device, num_classes, os.path.join(out_dir_path, 'outputs.test.csv'), 'testing_outputs')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    main(args)

