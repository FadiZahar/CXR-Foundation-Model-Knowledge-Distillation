import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.io import imread

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T



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
        try:
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
        
        except Exception as e:
            print(f"Error loading data for item {item}: {e}")
            raise Exception(f"Failed to load data for item {item}") from e

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
