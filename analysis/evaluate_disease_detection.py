import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import auc, recall_score, roc_auc_score, roc_curve
from sklearn.utils import resample
from tabulate import tabulate
from tqdm import tqdm
from pprint import pprint

# Import global shared variables
from config.config_shared import TARGET_FPR, N_BOOTSTRAP, CI_LEVEL, LABELS, RACES, SEXES
# Import the configuration loader
from config.loader_config import load_config, get_dataset_name

np.random.seed(42)



def bootstrap_ci(targets: np.ndarray, probs: np.ndarray, races: np.ndarray, sexes: np.ndarray, 
                 n_bootstrap: int = 2000, target_fpr: float = 0.2):
    n_samples = targets.shape[0]
    metrics = defaultdict(lambda: defaultdict(list))
    aucroc_metrics = defaultdict(dict)

    for n in tqdm(range(n_bootstrap + 1), desc='Bootstrap sampling'):
        # No resampling for the first iteration
        idx = resample(np.arange(n_samples), stratify=targets) if n > 0 else np.arange(n_samples)
        sample_targets, sample_probs = targets[idx], probs[idx]

        fpr, tpr, thres = roc_curve(sample_targets, sample_probs)
        optimal_index_target_fpr = np.argmin(np.abs(fpr - target_fpr))
        optimal_threshold = thres[optimal_index_target_fpr]

        # Compute and store metrics for 'all' population
        update_metrics(metrics=metrics, aucroc_metrics=aucroc_metrics if n == 0 else None, group_name="All", 
                       group_targets=sample_targets, group_probs=sample_probs, optimal_thresh=optimal_threshold)

        # Compute and store metrics for 'race' subgroups
        sample_races = races[idx]
        for race in RACES:
            race_subgroup_idx = (sample_races == race)
            race_targets, race_probs = sample_targets[race_subgroup_idx], sample_probs[race_subgroup_idx]
            update_metrics(metrics=metrics, aucroc_metrics=aucroc_metrics if n == 0 else None, group_name=race, 
                           group_targets=race_targets, group_probs=race_probs, optimal_thresh=optimal_threshold)
            
        # Compute and store metrics for 'sex' subgroups
        sample_sexes = sexes[idx]
        for sex in SEXES:
            sex_subgroup_idx = (sample_sexes == sex)
            sex_targets, sex_probs = sample_targets[sex_subgroup_idx], sample_probs[sex_subgroup_idx]
            update_metrics(metrics=metrics, aucroc_metrics=aucroc_metrics if n == 0 else None, group_name=sex, 
                           group_targets=sex_targets, group_probs=sex_probs, optimal_thresh=optimal_threshold)

    return metrics, aucroc_metrics


def update_metrics(metrics, aucroc_metrics, group_name, group_targets, group_probs, optimal_thresh):
    if len(group_targets) > 0:
        # Calculate metrics of interest:
        fpr, tpr, thres = roc_curve(group_targets, group_probs)
        auc_roc = roc_auc_score(group_targets, group_probs)
        binary_predictions = group_probs >= optimal_thresh
        tpr_global_thres = recall_score(group_targets, binary_predictions, pos_label=1)
        fpr_global_thres = 1 - recall_score(group_targets, binary_predictions, pos_label=0)

        # Update metrics:
        metrics["AUC-ROC"][group_name].append(auc_roc)
        metrics["TPR at threshold"][group_name].append(tpr_global_thres)
        metrics["FPR at threshold"][group_name].append(fpr_global_thres)
        metrics["Youden\'s Index"][group_name].append(tpr_global_thres - fpr_global_thres)

        # Update aucroc_metrics for the first iteration only (no resampling):
        if aucroc_metrics is not None:
            aucroc_metrics["TPRs"][group_name] = tpr
            aucroc_metrics["FPRs"][group_name] = fpr
            aucroc_metrics["TPR at threshold"][group_name] = tpr_global_thres
            aucroc_metrics["FPR at threshold"][group_name] = fpr_global_thres
            aucroc_metrics["AUC-ROC"][group_name] = auc_roc


def summarise_metrics(metrics, ci_level=0.95, return_ci=True):
    summary = {}
    alpha = (1 - ci_level) / 2
    for metric, groups in metrics.items():
        summary[metric] = {}
        for group, values in groups.items():
            flat_values = np.concatenate(values) if isinstance(values[0], np.ndarray) else values
            if return_ci:
                ci_lower = np.quantile(values, alpha)
                ci_upper = np.quantile(values, 1 - alpha)
                summary[metric][group] = f"{values[0]:.2f} ({ci_lower:.2f}-{ci_upper:.2f})"
            else:
                summary[metric][group] = np.mean(flat_values)
    return summary


def compute_relative_changes(df, metric_name="Youden\'s Index"):
    """Calculates the relative change from the 'all' average baseline as a percentage."""
    overall_avg = df.loc[metric_name, 'All'] 
    df_temp = df.T  # Transpose to make metrics columns
    df_temp[f"Relative Change (%) in {metric_name} from Average"] = ((df_temp[metric_name] - overall_avg) / overall_avg) * 100
    df_relative = df_temp[df_temp.index != 'All'].copy()
    return df_relative.T


def plot_metrics(results_df, label, output_dir, dataset_name, metric_name="Youden\'s Index", plot_type='absolute'):
    """Generates and saves plots based on the provided data."""
    plt.figure(figsize=(8, 5))
    y_col = metric_name if plot_type == 'absolute' else f"Relative Change (%) in {metric_name} from Average"
    title = f"{label} - {plot_type.capitalize()} Performance"
    
    # Transpose the DataFrame for plotting
    results_df = results_df.T

    # Create a color palette
    race_cmap = plt.get_cmap('Blues')
    sex_cmap = plt.get_cmap('Greens')
    all_cmap = plt.get_cmap('Oranges')
    # Generate colors from the colormap
    race_colors = [race_cmap(0.65 + i*0.25 / (len(RACES))) for i in range(len(RACES))]
    sex_colors = [sex_cmap(0.65 + i*0.25 / (len(SEXES))) for i in range(len(SEXES))]
    all_color = [all_cmap(0.65)]
    # Map colors to groups
    color_map = {group: race_colors[i] for i, group in enumerate(RACES)}
    color_map.update({group: sex_colors[i] for i, group in enumerate(SEXES)})
    color_map['All'] = all_color[0]
    palette = [color_map[group] for group in results_df.index if group in color_map]

    # Plot
    ax = sns.barplot(x=results_df.index, y=y_col, data=results_df, palette=palette, edgecolor='black', linewidth=1)
    # Adjust bar width
    bar_width = 0.4
    for bar in ax.patches:
        bar.set_width(bar_width)
        bar.set_x(bar.get_x() + bar_width/2)
    if plot_type == 'absolute':
        plt.ylim(0, 0.7)
    ax.axhline(0, color='black', linewidth=1.5) 
    ax.yaxis.grid(True, linestyle='--', which='major', color='gray', alpha=0.7)  

    plt.title(title)
    plt.ylabel(y_col)
    plt.xlabel('Subgroup')
    # plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}__{plot_type}_performance_plot__({label.replace(" ", "_")}).png'), dpi=300)
    plt.close()


def plot_auc_roc_curves(aucroc_metrics_df, label, output_dir, subgroups, lw=1.5, alpha=0.8, markersize=10):
    fig, ax = plt.subplots(figsize=(8, 5))
    original_cmap = plt.get_cmap('Dark2')
    # colors = [original_cmap(i) for i in range(original_cmap.N)]
    # colors[3] = 'crimson'  # Replace the fourth color pink (index 3)
    colors = list(original_cmap(np.linspace(0, 1, len(subgroups))))
    colors[2] = 'mediumvioletred'

    for idx, subgroup in enumerate(subgroups):
        fprs = aucroc_metrics_df.loc['FPRs', subgroup]
        tprs = aucroc_metrics_df.loc['TPRs', subgroup]
        auc_score = aucroc_metrics_df.loc['AUC-ROC', subgroup]
        plt.plot(fprs, tprs, lw=lw, alpha=alpha, label=f'{subgroup} AUC-ROC={auc_score:.2f}', color=colors[idx])

    plt.gca().set_prop_cycle(None)
        
    for idx, subgroup in enumerate(subgroups):
        # Plotting the global threshold point
        tpr_global_thresh = aucroc_metrics_df.loc['TPR at threshold', subgroup]
        fpr_global_thresh = aucroc_metrics_df.loc['FPR at threshold', subgroup]
        plt.plot(fpr_global_thresh, tpr_global_thresh, marker='X', linestyle='None', alpha=alpha, markersize=markersize,
                    label=f'{subgroup} TPR={tpr_global_thresh:.2f}, FPR={fpr_global_thresh:.2f}', color=colors[idx])

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='midnightblue', alpha=alpha)
    plt.annotate('Random Classifier', xy=(0.5, 0.54), fontsize=11, color='midnightblue', rotation=32)
    
    plt.xlabel('False Positive Rate (FPR)', fontsize=13)
    plt.ylabel('True Positive Rate (TPR)', fontsize=13)
    plt.title(f"{label} - ROC Curve", fontsize=14)
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
    plt.legend(loc='lower right', fontsize=11, ncol=2)
    ax.spines[['right', 'top']].set_visible(False)
    plt.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
    
    plt.savefig(os.path.join(output_dir, f'{dataset_name}__roc_curve__({label.replace(" ", "_")}).png'), dpi=300)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate performance metrics for classification model.")
    parser.add_argument('--outputs_dir', type=str, required=True, help='Path to outputs directory')
    parser.add_argument('--config', default='chexpert', choices=['chexpert', 'mimic'], help='Config dataset module to use')
    parser.add_argument('--labels', nargs='+', default=["No Finding", "Cardiomegaly", "Pneumothorax", "Pleural Effusion"], 
                        help='List of labels to process')
    parser.add_argument('--focus_metric', default="Youden\'s Index", help='Performance metric to focus on for comparative analysis')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    # Load the configuration dynamically based on the command line argument
    config = load_config(args.config)
    # Accessing the configuration to import dataset-specific variables
    dataset_name = get_dataset_name(args.config)
    TEST_RECORDS_CSV = config.TEST_RECORDS_CSV

    # Path to outputs and data characteristics files
    outputs_csv_file = os.path.join(args.outputs_dir, 'outputs_test.csv')
    model_outputs = pd.read_csv(outputs_csv_file)
    data_characteristics = pd.read_csv(TEST_RECORDS_CSV)

    # Evaluation parameters
    target_fpr = TARGET_FPR
    n_bootstrap = N_BOOTSTRAP
    ci_level = CI_LEVEL

    # Directories to save analysis outputs
    disease_detection_dir_path = os.path.join(args.outputs_dir, 'analysis', 'disease_detection')
    performance_tables_dir_path = os.path.join(disease_detection_dir_path, 'performance_tables/')
    absolute_plots_dir_path = os.path.join(disease_detection_dir_path, 'performance_plots_absolute/')
    relative_plots_dir_path = os.path.join(disease_detection_dir_path, 'performance_plots_relative/')
    aucroc_plots_dir_path = os.path.join(disease_detection_dir_path, 'aucroc_plots/')
    aucroc_tables_dir_path = os.path.join(disease_detection_dir_path, 'aucroc_tables/')
    os.makedirs(disease_detection_dir_path, exist_ok=True)
    os.makedirs(performance_tables_dir_path, exist_ok=True)
    os.makedirs(absolute_plots_dir_path, exist_ok=True)
    os.makedirs(relative_plots_dir_path, exist_ok=True)
    os.makedirs(aucroc_plots_dir_path, exist_ok=True)
    os.makedirs(aucroc_tables_dir_path, exist_ok=True)


    # Focus is on Youden's Index
    metric_name = args.focus_metric

    for label in args.labels:
        label_index = LABELS.index(label)
        probs = model_outputs[f"prob_class_{label_index+1}"]
        targets = np.array(model_outputs[f"target_class_{label_index+1}"])
        races = data_characteristics.race.values
        sexes = data_characteristics.sex.values

        results, aucroc_metrics = bootstrap_ci(targets=targets, probs=probs, races=races, sexes=sexes, 
                                               n_bootstrap=n_bootstrap, target_fpr=target_fpr)
        results_with_ci = summarise_metrics(results, ci_level, return_ci=True)
        results_plain_avg = summarise_metrics(results, ci_level, return_ci=False)

        columns_as_in_manuscript = [race for race in RACES] + [sex for sex in SEXES] + ["All"]
        results_df = pd.DataFrame.from_dict(results, orient="index")[columns_as_in_manuscript]
        aucroc_metrics_df = pd.DataFrame.from_dict(aucroc_metrics, orient="index")[columns_as_in_manuscript]
        results_with_ci_df = pd.DataFrame.from_dict(results_with_ci, orient="index")[columns_as_in_manuscript]
        results_plain_avg_df = pd.DataFrame.from_dict(results_plain_avg, orient="index")[columns_as_in_manuscript]

        results_df.to_csv(os.path.join(performance_tables_dir_path, f'{dataset_name}__all_performance_metrics__({label.replace(" ", "_")}).csv'), index=True)
        aucroc_metrics_df.to_csv(os.path.join(aucroc_tables_dir_path, f'{dataset_name}__all_aucroc_metrics__({label.replace(" ", "_")}).csv'), index=True)
        
        results_with_ci_df.to_csv(os.path.join(performance_tables_dir_path, f'{dataset_name}__ci_summary_performance_metrics__({label.replace(" ", "_")}).csv'), index=True)
        print(f"\nResults for: {label.upper()} ({ci_level * 100:.0f}%-CI with {n_bootstrap} bootstrap samples)")
        print(tabulate(results_with_ci_df, headers=results_with_ci_df.columns))

        # Plot Absolute Performance
        plot_metrics(results_df=results_plain_avg_df, label=label, dataset_name=dataset_name, metric_name=metric_name, 
                     output_dir=absolute_plots_dir_path, plot_type='absolute')

        # Compute and plot Relative Performance Changes
        relative_results_df = compute_relative_changes(df=results_plain_avg_df, metric_name=metric_name)
        plot_metrics(results_df=relative_results_df, label=label, dataset_name=dataset_name, metric_name=metric_name, 
                     output_dir=relative_plots_dir_path, plot_type='relative')
        
        # Plot AUC-ROC curves 
        plot_auc_roc_curves(aucroc_metrics_df=aucroc_metrics_df, label=label, output_dir=aucroc_plots_dir_path, 
                            subgroups=columns_as_in_manuscript)


