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
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as T
from torchvision import models
from torchmetrics import MultilabelAccuracy

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar



image_size = (224, 224)
CXRFM_embeds_size = 1376
batch_size = 150
learning_rate = 0.001
epochs = 20
num_workers = 4

cxrs_filepath = '/vol/biodata/data/chest_xray/CheXpert-v1.0/'
embeddings_filepath = '/vol/biomedic3/bglocker/cxr-foundation/outputs/chexpert/cxr_numpy/'
train_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.train.csv'
val_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.val.csv'
# test_records_csv = '/vol/biodata/data/chest_xray/CheXpert-v1.0/meta/algorithmic_encoding/chexpert.sample.test.csv' --> should be reserved for downstream task fine-tuning
main_dir_path = '/vol/biomedic3/bglocker/mscproj24/fz221/outputs/'
out_dir_name = 'CXR-FMKD_KD-initialisation/'



class CheXpertDataset(Dataset):
    def __init__(self, image_size: tuple, records_filepath: str, cxrs_filepath: str, embeddings_filepath: str, 
                 augmentation: bool = False, pseudo_rgb: bool = True):
        self.image_size = image_size
        self.entries = pd.read_csv(records_filepath)
        self.cxrs_filepath = cxrs_filepath
        self.embeddings_filepath = embeddings_filepath
        self.do_augment = augmentation
        self.pseudo_rgb = pseudo_rgb

        self.augment = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(transforms=[T.RandomAffine(degrees=15, scale=(0.9, 1.1))], p=0.5),
        ])
        
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
        image_filepath = record['image_filepath']

        # Get the Chest X-Rays (cxrs)
        cxr_filepath = os.path.join(self.cxrs_filepath, image_filepath)   # e.g., '<cxrs_filepath>/preproc_224x224/patient24428_study22_view1_frontal.jpg'
        cxr = imread(cxr_filepath).astype(np.float32)   # cxr: Chest X-Ray

        # Get the corresponding CXR-FM embeddings
        patient_filename = os.path.basename(image_filepath)   # e.g., 'patient24428_study22_view1_frontal.jpg'
        embedding_filepath = os.path.join(self.embeddings_filepath, patient_filename.replace('.jpg', '.dat'))   # e.g., '<embeddings_filepath>/patient24428_study22_view1_frontal.dat'
        embedding = np.fromfile(embedding_filepath, dtype=np.float32)

        return {'cxr': cxr, 'embedding': embedding, 'label': record['label']}

    def __getitem__(self, item):
        sample = self.get_sample(item)

        cxr_tensor = torch.from_numpy(sample['cxr']).unsqueeze(0)
        embedding_tensor = torch.from_numpy(sample['embedding'])
        label_tensor = torch.from_numpy(sample['label'])

        if self.do_augment:
            cxr_tensor = self.augment(cxr_tensor)

        if self.pseudo_rgb:
            cxr_tensor = cxr_tensor.repeat(3, 1, 1)

        return {'cxr': cxr_tensor, 'embedding': embedding_tensor, 'label': label_tensor}

    def __len__(self):
        return len(self.entries)


class CheXpertDataModule(LightningDataModule):
    def __init__(self, image_size: tuple, train_records: str, val_records: str, cxrs_filepath: str, embeddings_filepath: str, 
                 pseudo_rgb: bool, batch_size: int, num_workers: int):
        super().__init__()
        self.image_size = image_size

        self.train_records = train_records
        self.val_records = val_records

        self.cxrs_filepath = cxrs_filepath
        self.embeddings_filepath = embeddings_filepath
        self.pseudo_rgb = pseudo_rgb

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = CheXpertDataset(image_size=self.image_size, records_filepath=self.train_records, cxrs_filepath=self.cxrs_filepath,
                                         embeddings_filepath=self.embeddings_filepath, augmentation=True, pseudo_rgb=self.pseudo_rgb)
        dev_set = CheXpertDataset(image_size=self.image_size, records_filepath=self.val_records, cxrs_filepath=self.cxrs_filepath,
                                       embeddings_filepath=self.embeddings_filepath, augmentation=False, pseudo_rgb=self.pseudo_rgb)
        self.val_set, self.test_set = random_split(dev_set, [0.7, 0.3])

        print('>> train_set size: ', len(self.train_set))
        print('>> val_set size:   ', len(self.val_set))
        print('>> test_set size:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class Pre_CXR_FMKD(LightningModule):
    def __init__(self, learning_rate: float, embedding_size: int):
        super().__init__()
        self.learning_rate = learning_rate
        self.embedding_size = embedding_size
        
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
                              train_records=train_records_csv,
                              val_records=val_records_csv,
                              cxrs_filepath=cxrs_filepath,
                              embeddings_filepath=embeddings_filepath,
                              pseudo_rgb=True,
                              batch_size=batch_size,
                              num_workers=num_workers)

    # Model
    model_type = Pre_CXR_FMKD
    model = model_type(learning_rate=learning_rate, embedding_size=CXRFM_embeds_size)

    # Create output directory
    out_dir_path = os.path.join(main_dir_path, out_dir_name)
    os.makedirs(out_dir_path, exist_ok=True)
    # Create TensorBoard logs directory
    logs_dir_path = os.path.join(out_dir_path, 'lightning_logs/')
    os.makedirs(logs_dir_path, exist_ok=True)
    # Create a temp. directory
    temp_dir_path = os.path.join(out_dir_path, 'temp')
    os.makedirs(temp_dir_path, exist_ok=True)
    for idx in range(5):
        sample = data.train_set.get_sample(idx)
        imsave(os.path.join(temp_dir_path, 'sample_' + str(idx) + '.jpg'), sample['cxr'].astype(np.uint8))

    # Train
    trainer = Trainer(
        callbacks=[ModelCheckpoint(monitor='val_loss', mode='min'), TQDMProgressBar(refresh_rate=10)],
        log_every_n_steps=5,
        max_epochs=epochs,
        accelerator='auto',
        devices=hparams.gpus,
        logger=TensorBoardLogger(logs_dir_path, name=out_dir_name),
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

