import os
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Import global variables
from config.config_chexpert import LABELS



class MetricTracker:
    def __init__(self, labels):
        self.labels = labels
        # Initialise dictionaries to store metrics as lists for each class and for macro metrics
        self.roc_auc_per_class = [[] for _ in labels]
        self.pr_auc_per_class = [[] for _ in labels]
        self.j_index_max_per_class = [[] for _ in labels]
        self.j_index_fpr_per_class = [[] for _ in labels]
        self.roc_auc_macro = []
        self.pr_auc_macro = []


    def update(self, roc_auc_per_class, roc_auc_macro, 
               pr_auc_per_class, pr_auc_macro, 
               j_indices_max, j_indices_target_fpr):
        # Update metric lists with new values from current epoch
        for i in range(len(self.labels)):
            self.roc_auc_per_class[i].append(roc_auc_per_class[i])
            self.pr_auc_per_class[i].append(pr_auc_per_class[i])
            self.j_index_max_per_class[i].append(j_indices_max[i])
            self.j_index_fpr_per_class[i].append(j_indices_target_fpr[i])
        # Update macro metrics
        self.roc_auc_macro.append(roc_auc_macro)
        self.pr_auc_macro.append(pr_auc_macro)


    def log_to_wandb(self, out_dir_path, phase, target_fpr):
        # Plot and log ROC-AUC per class and macro ROC-AUC
        self.plot_metrics(f"ROC-AUC per Class ({phase} phase)", self.roc_auc_per_class, f"ROC-AUC", 'roc', out_dir_path, phase)
        wandb.log({f"Macro ROC-AUC ({phase} phase)": self.roc_auc_macro})
        # Plot and log PR-AUC per class and macro PR-AUC
        self.plot_metrics(f"PR-AUC per Class ({phase} phase)", self.pr_auc_per_class, f"PR-AUC", 'pr', out_dir_path, phase)
        wandb.log({f"Macro PR-AUC ({phase} phase)": self.pr_auc_macro})
        # Plot and log Youden's Index per class for max and target FPR
        self.plot_metrics(f"Youden Index Max per Class ({phase} phase)", self.j_index_max_per_class, 
                        f"J-Index Max", 'yim', out_dir_path, phase)
        self.plot_metrics(f"Youden Index @ {target_fpr} FPR per Class ({phase} phase)", self.j_index_fpr_per_class, 
                        f"J-Index @ {target_fpr} FPR", 'yi', out_dir_path, phase)


    def plot_metrics(self, title, data, metric_name, palette_type, out_dir_path, phase):
        if palette_type == 'roc':
            color_palette = plt.cm.viridis(np.linspace(0, 1, len(data))) 
        elif palette_type == 'pr':
            color_palette = plt.cm.plasma(np.linspace(0, 1, len(data))) 
        elif palette_type == 'yi':
            color_palette = plt.cm.twilight_shifted(np.linspace(0, 1, len(data)))
        else:
            color_palette = plt.cm.copper(np.linspace(0, 1, len(data)))

        epochs = list(range(1, len(data[0]) + 1))
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for idx, class_data in enumerate(data):
            plt.plot(epochs, class_data, label=f'Class {idx + 1} [{LABELS[idx]}]', color=color_palette[idx],
                    marker='o', linestyle='-' if len(class_data) > 1 else '')

        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.xticks(epochs) 
        plt_path = os.path.join(out_dir_path, f'{metric_name}_over_epochs_({phase}).png')
        plt.savefig(plt_path, bbox_inches='tight') 
        wandb.log({title: wandb.Image(fig)})
        plt.close(fig)



def calculate_roc_auc(targets, probs):
    """Calculate ROC AUC scores."""
    if not targets.shape == probs.shape:
        raise ValueError("Shape of targets and probs must match.")
    
    roc_auc_per_class = roc_auc_score(targets, probs, average=None)
    roc_auc_macro = roc_auc_score(targets, probs, average="macro")
    return roc_auc_per_class, roc_auc_macro


def calculate_pr_auc(targets, probs):
    """Calculate Precision-Recall (PR) AUC scores."""
    if not targets.shape == probs.shape:
        raise ValueError("Shape of targets and probs must match.")
    
    pr_auc_per_class = average_precision_score(targets, probs, average=None)
    pr_auc_macro = average_precision_score(targets, probs, average="macro")
    return pr_auc_per_class, pr_auc_macro


def calculate_youden_index(targets, probs, target_fpr):
    """Calculate Youden's J index for optimal cutoff and at a specific FPR."""
    if not isinstance(target_fpr, (int, float)) or not (0 <= target_fpr <= 1):
        raise ValueError("target_fpr must be between 0 and 1.")
    if not targets.shape == probs.shape:
        raise ValueError("Shape of targets and probs must match.")
    
    j_indices_max = []
    j_indices_target_fpr = []
    
    for i in range(targets.shape[1]):
        fpr, tpr, thresholds = roc_curve(targets[:, i], probs[:, i])
        j_index = tpr - fpr
        max_j_index = np.max(j_index)
        j_indices_max.append(max_j_index)
        
        # Find the closest index where FPR is less than or equal to target_fpr
        idx_target = np.where(fpr <= target_fpr)[0][-1]
        j_index_at_target_fpr = tpr[idx_target] - fpr[idx_target]
        j_indices_target_fpr.append(j_index_at_target_fpr)
    
    return j_indices_max, j_indices_target_fpr



# Initialise the tracker
metric_tracker = MetricTracker(LABELS)


def generate_and_log_metrics(targets, probs, out_dir_path, phase, target_fpr=0.2):
    roc_auc_per_class, roc_auc_macro = calculate_roc_auc(targets, probs)
    pr_auc_per_class, pr_auc_macro = calculate_pr_auc(targets, probs)
    j_indices_max, j_indices_target_fpr = calculate_youden_index(targets, probs, target_fpr)

    metric_tracker.update(roc_auc_per_class=roc_auc_per_class, roc_auc_macro=roc_auc_macro, 
                          pr_auc_per_class=pr_auc_per_class, pr_auc_macro=pr_auc_macro, 
                          j_indices_max=j_indices_max, j_indices_target_fpr=j_indices_target_fpr)
    metric_tracker.log_to_wandb(out_dir_path=out_dir_path, phase=phase, target_fpr=target_fpr)



