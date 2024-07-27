import numpy as np
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Import global variables
from config.config_chexpert import LABELS



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


def generate_and_log_metrics(targets, probs, target_fpr=0.2):
    roc_auc_per_class, roc_auc_macro = calculate_roc_auc(targets, probs)
    pr_auc_per_class, pr_auc_macro = calculate_pr_auc(targets, probs)
    j_indices_max, j_indices_target_fpr = calculate_youden_index(targets, probs, target_fpr)

    # Preparing data for logging
    roc_auc_logs = {f"ROC-AUC for Class {i+1} [{LABELS[i]}]": roc_auc for i, roc_auc in enumerate(roc_auc_per_class)}
    pr_auc_logs = {f"PR-AUC for Class {i+1} [{LABELS[i]}]": pr_auc for i, pr_auc in enumerate(pr_auc_per_class)}
    j_max_logs = {f"Youden Index (Max) for Class {i+1} [{LABELS[i]}]": j_idx for i, j_idx in enumerate(j_indices_max)}
    j_target_fpr_logs = {f"Youden Index at FPR={target_fpr} for Class {i+1} [{LABELS[i]}]": j_idx for i, j_idx in enumerate(j_indices_target_fpr)}

    # Logging ROC-AUC
    wandb.log(roc_auc_logs)
    wandb.log({"ROC-AUC (Macro)": roc_auc_macro})

    # Logging PR-AUC
    wandb.log(pr_auc_logs)
    wandb.log({"PR-AUC (Macro)": pr_auc_macro})

    # Logging Youden's Indices
    wandb.log(j_max_logs)
    wandb.log(j_target_fpr_logs)


