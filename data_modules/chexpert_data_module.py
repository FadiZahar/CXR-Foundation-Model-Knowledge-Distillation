from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule
from typing import List, Optional

# Import custom modules
from data_modules.chexpert_dataset import CheXpertDataset



class CheXpertDataModule(LightningDataModule):
    def __init__(self, image_size: tuple, cxrs_filepath: str, embeddings_filepath: str, pseudo_rgb: bool, batch_size: int, num_workers: int, 
                 train_records: str, val_records: str, test_records: Optional[str] = None, dev_split: Optional[List[float]] = None):
        super().__init__()
        self.image_size = image_size

        self.train_records = train_records
        self.val_records = val_records
        self.test_records = test_records
        self.dev_split = dev_split if dev_split is not None else [0.7, 0.3]

        self.cxrs_filepath = cxrs_filepath
        self.embeddings_filepath = embeddings_filepath
        self.pseudo_rgb = pseudo_rgb

        self.batch_size = batch_size
        self.num_workers = num_workers


        self.train_set = CheXpertDataset(image_size=self.image_size, records_filepath=self.train_records, cxrs_filepath=self.cxrs_filepath,
                                         embeddings_filepath=self.embeddings_filepath, augmentation=True, pseudo_rgb=self.pseudo_rgb)
        if self.test_records:
            self.val_set = CheXpertDataset(image_size=self.image_size, records_filepath=self.val_records, cxrs_filepath=self.cxrs_filepath,
                                        embeddings_filepath=self.embeddings_filepath, augmentation=False, pseudo_rgb=self.pseudo_rgb)
            self.test_set = CheXpertDataset(image_size=self.image_size, records_filepath=self.test_records, cxrs_filepath=self.cxrs_filepath,
                                        embeddings_filepath=self.embeddings_filepath, augmentation=False, pseudo_rgb=self.pseudo_rgb)
        else:
            dev_set = CheXpertDataset(image_size=self.image_size, records_filepath=self.val_records, cxrs_filepath=self.cxrs_filepath,
                                       embeddings_filepath=self.embeddings_filepath, augmentation=False, pseudo_rgb=self.pseudo_rgb)
            self.val_set, self.test_set = random_split(dev_set, self.dev_split)

        print('>> train_set size: ', len(self.train_set))
        print('>> val_set size:   ', len(self.val_set))
        print('>> test_set size:  ', len(self.test_set))
        

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)