import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

# Import global variables
from config.config_shared import LABELS



class MetricTracker:
    def __init__(self, labels):
        self.labels = labels
        self.target_fpr = 0.2
        self.current_phase = None
        self.epoch_offset = 0 
        self.ordered_epoch_offsets = []
        self.ordered_phases = []
        self.reset_metrics()


    def reset_metrics(self):
        self.roc_auc_per_class = [[] for _ in self.labels]
        self.pr_auc_per_class = [[] for _ in self.labels]
        self.j_index_max_per_class = [[] for _ in self.labels]
        self.j_index_fpr_per_class = [[] for _ in self.labels]
        self.roc_auc_macro = []
        self.pr_auc_macro = []

    
    def check_phase(self, phase):
        if self.current_phase != phase:
            self.current_phase = phase
            self.epoch_offset = len(self.pr_auc_macro)
            self.ordered_epoch_offsets.append(self.epoch_offset)
            self.ordered_phases.append(phase)
    

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
        self.plot_metrics(f"AUC-ROC per Class ({phase})", self.roc_auc_per_class, f"AUC-ROC Score", 'roc', out_dir_path, phase)
        current_roc_auc_macro = self.roc_auc_macro[-1]
        wandb.log({f"Macro AUC-ROC ({phase})": current_roc_auc_macro})

        # Plot and log PR-AUC per class and macro PR-AUC
        self.plot_metrics(f"AUC-PR per Class ({phase})", self.pr_auc_per_class, f"AUC-PR Score", 'pr', out_dir_path, phase)
        current_pr_auc_macro = self.pr_auc_macro[-1]
        wandb.log({f"Macro AUC-PR ({phase})": current_pr_auc_macro})

        # Plot and log Youden's Index per class for max and target FPR
        self.plot_metrics(f"Max Youden's Index per Class ({phase})", self.j_index_max_per_class, 
                        f"Max Youden's Index", 'yim', out_dir_path, phase)
        self.plot_metrics(f"Youden's Index at {int(target_fpr*100)}% FPR per Class ({phase})", self.j_index_fpr_per_class, 
                        f"Youden's Index at Target FPR", 'yi', out_dir_path, phase)


    def plot_metrics(self, title, data, metric_name, palette_type, out_dir_path, phase):
        color_map = {
            'roc': plt.cm.viridis,
            'pr': plt.cm.plasma,
            'yi': plt.cm.twilight_shifted,
            'default': plt.cm.copper
        }
        color_palette = color_map.get(palette_type, color_map['default'])(np.linspace(0, 1, len(data)))

        phase_index = self.ordered_phases.index(phase)
        start_epoch = self.ordered_epoch_offsets[phase_index]
        end_epoch = self.ordered_epoch_offsets[phase_index + 1] if (phase_index + 1) < len(self.ordered_epoch_offsets) else len(data[0])
        epochs = list(range(end_epoch - start_epoch))  # Start epochs at 0 (convention)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for idx, class_data in enumerate(data):
            plt.plot(epochs, class_data[start_epoch:end_epoch], label=f'Class {idx + 1} [{LABELS[idx]}]', color=color_palette[idx],
                    marker='o', linestyle='-' if len(class_data) > 1 else '')

        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(True)
        plt.xticks(epochs) 
        self.save_plot(fig, title, out_dir_path)


    def save_plot(self, fig, title, out_dir_path):
        # Directories for high and low resolution images
        high_res_dir = os.path.join(out_dir_path, 'metrics_plots', 'high_res')
        low_res_dir = os.path.join(out_dir_path, 'metrics_plots', 'low_res')
        os.makedirs(high_res_dir, exist_ok=True)
        os.makedirs(low_res_dir, exist_ok=True)

        # Paths for high and low resolution images
        high_res_path = os.path.join(high_res_dir, f'{title}.png')
        low_res_path = os.path.join(low_res_dir, f'{title}.png')

        # Save high-resolution image
        plt.savefig(high_res_path, bbox_inches='tight', dpi=300)

        # Save low-resolution image for logging
        plt.savefig(low_res_path, bbox_inches='tight')
        wandb.log({title: wandb.Image(low_res_path)})

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
        # idx_target = np.where(fpr <= target_fpr)[0][-1]

        # Find the closest index to the target_fpr
        idx_target = np.argmin(np.abs(fpr - target_fpr))
        
        j_index_at_target_fpr = tpr[idx_target] - fpr[idx_target]
        j_indices_target_fpr.append(j_index_at_target_fpr)
    
    return j_indices_max, j_indices_target_fpr



# Initialise the tracker
metric_tracker = MetricTracker(LABELS)


def generate_and_log_metrics(targets, probs, out_dir_path, phase, target_fpr=0.2):
    metric_tracker.target_fpr = target_fpr
    metric_tracker.check_phase(phase)
    roc_auc_per_class, roc_auc_macro = calculate_roc_auc(targets, probs)
    pr_auc_per_class, pr_auc_macro = calculate_pr_auc(targets, probs)
    j_indices_max, j_indices_target_fpr = calculate_youden_index(targets, probs, target_fpr)

    metric_tracker.update(roc_auc_per_class=roc_auc_per_class, roc_auc_macro=roc_auc_macro, 
                          pr_auc_per_class=pr_auc_per_class, pr_auc_macro=pr_auc_macro, 
                          j_indices_max=j_indices_max, j_indices_target_fpr=j_indices_target_fpr)
    metric_tracker.log_to_wandb(out_dir_path=out_dir_path, phase=phase, target_fpr=target_fpr)


def save_and_plot_all_metrics(out_dir_path):
    plot_all_metrics(f"AUC-ROC per Class (All Phases)", metric_tracker.roc_auc_per_class, f"AUC-ROC Score", 'roc', out_dir_path)
    plot_all_metrics(f"AUC-PR per Class (All Phases)", metric_tracker.pr_auc_per_class, f"AUC-PR Score", 'pr', out_dir_path)
    plot_all_metrics(f"Max Youden's Index per Class (All Phases)", metric_tracker.j_index_max_per_class, 
                        f"Max Youden's Index", 'yim', out_dir_path)
    plot_all_metrics(f"Youden's Index at {int(metric_tracker.target_fpr*100)}% FPR per Class (All Phases)", metric_tracker.j_index_fpr_per_class, 
                        f"Youden's Index at Target FPR", 'yi', out_dir_path)
    save_all_metrics(out_dir_path=out_dir_path)


def save_all_metrics(out_dir_path):
    metrics_path = os.path.join(out_dir_path, 'metrics_csv')
    os.makedirs(metrics_path, exist_ok=True)

    for phase_index, phase in enumerate(metric_tracker.ordered_phases):
        csv_path = os.path.join(metrics_path, f"{phase}.csv")
        start_epoch = metric_tracker.ordered_epoch_offsets[phase_index]
        if (phase_index + 1) < len(metric_tracker.ordered_epoch_offsets):
            end_epoch = metric_tracker.ordered_epoch_offsets[phase_index + 1]
        else:
            end_epoch = len(metric_tracker.roc_auc_macro)
        
        with open(csv_path, 'w', newline='') as file:
            fieldnames = ['epoch']  # Start with the epoch column
            # Extend fieldnames for each metric
            for idx in range(len(metric_tracker.labels)):
                fieldnames.extend([
                    f'auc_roc_class{idx+1}',
                    f'auc_pr_class{idx+1}',
                    f'j_index_max_class{idx+1}',
                    f'j_index_fpr_class{idx+1}'
                ])
            fieldnames.extend(['macro_auc_roc', 'macro_auc_pr'])
            
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            
            # Write data for epochs in the current phase
            for epoch in range(start_epoch, end_epoch):
                adjusted_epoch = epoch - start_epoch
                row = {'epoch': adjusted_epoch}
                for idx in range(len(metric_tracker.labels)):
                    row[f'auc_roc_class{idx+1}'] = metric_tracker.roc_auc_per_class[idx][epoch]
                    row[f'auc_pr_class{idx+1}'] = metric_tracker.pr_auc_per_class[idx][epoch]
                    row[f'j_index_max_class{idx+1}'] = metric_tracker.j_index_max_per_class[idx][epoch]
                    row[f'j_index_fpr_class{idx+1}'] = metric_tracker.j_index_fpr_per_class[idx][epoch]
                row['macro_auc_roc'] = metric_tracker.roc_auc_macro[epoch]
                row['macro_auc_pr'] = metric_tracker.pr_auc_macro[epoch]
                writer.writerow(row)


def plot_all_metrics(title, data, metric_name, palette_type, out_dir_path):
    color_map = {
        'roc': plt.cm.viridis,
        'pr': plt.cm.plasma,
        'yi': plt.cm.twilight_shifted,
        'default': plt.cm.copper
    }
    color_palette = color_map.get(palette_type, color_map['default'])(np.linspace(0, 1, len(data)))

    epochs = list(range(len(data[0])))  # Full range of epochs, starting at 0
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for idx, class_data in enumerate(data):
        plt.plot(epochs, class_data, label=f'Class {idx + 1} [{LABELS[idx]}]', color=color_palette[idx],
                marker='o', linestyle='-' if len(class_data) > 1 else '')

    # Adding vertical lines for phase transitions
    line_styles = ['--', '-.', ':']
    line_width = 2
    phase_colors = ['black', 'red', 'blue', 'green', 'purple']
    for phase_index, phase_offset in enumerate(metric_tracker.ordered_epoch_offsets):  # Skip the first offset
        style = line_styles[phase_index % len(line_styles)]
        color = phase_colors[phase_index // len(line_styles) % len(phase_colors)]
        plt.axvline(x=phase_offset, color=color, linestyle=style, linewidth=line_width,
                    label=f'Start of phase: {metric_tracker.ordered_phases[phase_index]}')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.xticks(epochs) 
    metric_tracker.save_plot(fig, title, out_dir_path)



