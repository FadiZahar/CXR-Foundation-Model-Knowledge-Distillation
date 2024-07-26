from pytorch_lightning.callbacks import Callback
import pandas as pd
import torch
import csv



class EvalLoggingCallback(Callback):
    def __init__(self, num_classes, file_path):
        self.num_classes = num_classes
        self.file_path = file_path
        self.reset_storage()

    def reset_storage(self):
        self.logits_list = []
        self.probs_list = []
        self.targets_list = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        logits = outputs.get('logits')
        probs = torch.sigmoid(logits)
        labels = batch['label']

        self.logits_list.append(logits)
        self.probs_list.append(probs)
        self.targets_list.append(labels)

    def on_test_end(self, trainer, pl_module):
        logits_array = torch.cat(self.logits_list, dim=0)
        probs_array = torch.cat(self.probs_list, dim=0)
        targets_array = torch.cat(self.targets_list, dim=0)

        counts = []
        for i in range(self.num_classes):
            t = (targets_array[:, i] == 1)
            c = torch.sum(t).item()
            counts.append(c)
        print("Class counts:", counts)

        df_logits = pd.DataFrame(logits_array.cpu().numpy(), columns=[f'logit_class_{i+1}' for i in range(self.num_classes)])
        df_probs = pd.DataFrame(probs_array.cpu().numpy(), columns=[f'prob_class_{i+1}' for i in range(self.num_classes)])
        df_targets = pd.DataFrame(targets_array.cpu().numpy(), columns=[f'target_class_{i+1}' for i in range(self.num_classes)])
        
        df = pd.concat([df_logits, df_probs, df_targets], axis=1)
        df.to_csv(self.file_path, index=False)

        self.reset_storage() 

