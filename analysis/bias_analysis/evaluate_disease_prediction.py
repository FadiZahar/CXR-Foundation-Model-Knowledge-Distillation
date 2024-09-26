import os
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import recall_score, roc_auc_score, roc_curve
from sklearn.utils import resample
from tabulate import tabulate
from tqdm import tqdm
import pickle

from matplotlib import font_manager
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.lines as mlines
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerBase
import matplotlib.gridspec as gridspec


# Import global shared variables
from config.config_shared import TARGET_FPR, N_BOOTSTRAP, CI_LEVEL, OUT_DPI, LABELS, RACES, SEXES
# Import the configuration loader
from config.loader_config import load_config, get_dataset_name

# Check if Latin Modern Roman (~LaTeX) is available, and set it; otherwise, use the default font
if 'Latin Modern Roman' in [f.name for f in font_manager.fontManager.ttflist]:
    plt.rcParams['font.family'] = 'Latin Modern Roman'

# Global Variables
BAR_EDGE_COLOUR = (0.45, 0.45, 0.45)
ERROR_EDGE_COLOUR = (0.25, 0.25, 0.25)
LINE_WIDTH=0.8

np.random.seed(42)




# ========================================================
# ==== PERFORMANCE ANALYSIS - UTILS FUNCTIONS - START ====
# ========================================================

## Section for plot generation:

class DiagonalColorHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Extract the two colors stored in the custom attribute of the original legend handle
        color1, color2 = orig_handle.custom_colors 
        patch = patches.FancyBboxPatch((xdescent, ydescent), width, height, 
                                       boxstyle="square,pad=-1", facecolor=color1, edgecolor=BAR_EDGE_COLOUR, linewidth=LINE_WIDTH, transform=trans)
        patch.set_clip_path(patches.Polygon([[xdescent, ydescent + height], [xdescent + width, ydescent], 
                                             [xdescent + width, ydescent + height]], closed=True, transform=trans))
        patch2 = patches.FancyBboxPatch((xdescent, ydescent), width, height, 
                                        boxstyle="square,pad=-1", facecolor=color2, edgecolor=BAR_EDGE_COLOUR, linewidth=LINE_WIDTH, transform=trans)
        patch2.set_clip_path(patches.Polygon([[xdescent, ydescent], [xdescent + width, ydescent], 
                                              [xdescent, ydescent + height]], closed=True, transform=trans))
        return [patch, patch2]


def set_plot_labels(ax, label, metric, plot_type, font_size, title_pad, label_pad, title_font_delta, axis_font_delta, dataset_name):
    if plot_type == 'absolute':
        if "AUC-ROC" in metric:
            ax.set_title(f"AUC-ROC (Absolute) — {label} | {dataset_name}", fontsize=font_size+title_font_delta, pad=title_pad)
            ax.set_ylabel("AUC-ROC", fontsize=font_size+axis_font_delta, labelpad=label_pad)
        elif "Youden's Index" in metric:
            ax.set_title(f"Youden's J Statistic (Absolute) — {label} | {dataset_name}", fontsize=font_size+title_font_delta, pad=title_pad)
            ax.set_ylabel("Youden's J Statistic", fontsize=font_size+axis_font_delta, labelpad=label_pad)
    else:
        if "AUC-ROC" in metric:
            ax.set_title(f"AUC-ROC (Relative) — {label} | {dataset_name}", fontsize=font_size+title_font_delta, pad=title_pad)
            ax.set_ylabel("Difference from Average (%)\nAUC-ROC", fontsize=font_size+axis_font_delta, labelpad=label_pad)
        elif "Youden's Index" in metric:
            ax.set_title(f"Youden's J Statistic (Relative) — {label} | {dataset_name}", fontsize=font_size+title_font_delta, pad=title_pad)
            ax.set_ylabel("Difference from Average (%)\nYouden's J Statistic", fontsize=font_size+axis_font_delta, labelpad=label_pad)


def get_plot_title(metric, plot_type, dataset_name):
    if plot_type == 'absolute':
        if "AUC-ROC" in metric:
            title = f"AUC-ROC (Absolute) | {dataset_name}"
        elif "Youden's Index" in metric:
            title = f"Youden's J Statistic (Absolute) | {dataset_name}"
    else:
        if "AUC-ROC" in metric:
            title = f"AUC-ROC (Relative) | {dataset_name}"
        elif "Youden's Index" in metric:
            title = f"Youden's J Statistic (Relative) | {dataset_name}"
    return title  


def create_custom_legend(ax, models, full_cmap, plot_type, bar_edge_colour, error_bar_colour, font_size, line_width):
    legend_elements = []
    for i in range(len(models)):
        if plot_type == 'absolute':
            # Use diagonal colors for the absolute plot type
            patch = patches.Patch(label=models[i]["shortname"])
            patch.custom_colors = (full_cmap[i*4+3], full_cmap[i*4+2])
        else:
            # Use a single color for non-absolute plot types
            patch = patches.Patch(label=models[i]["shortname"], facecolor=full_cmap[i*4+3], 
                                edgecolor=bar_edge_colour, linewidth=line_width)
        legend_elements.append(patch)

    # Add a custom legend entry for error bars (95% CI)
    error_bar_legend = mlines.Line2D([], [], color=error_bar_colour, marker='o', linestyle='-', markersize=4, 
                                    label='95% CI (2000 Bootstrap Samples)', markerfacecolor=error_bar_colour, 
                                    markeredgewidth=1, markeredgecolor=error_bar_colour, linewidth=line_width)

    # Combine both custom legend elements
    legend_elements.append(error_bar_legend)

    # Legend is made out of 6 elements, the last 3 are lengthy in writing: use 3 columns (i.e., 2 rows)
    num_columns = 3

    # Update the legend with custom elements
    ax.legend(handles=legend_elements, 
              handler_map={patches.Patch: DiagonalColorHandler()} if plot_type == 'absolute' else None,
              loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=font_size-1, ncol=num_columns)

    # Extract handles and labels from the legend_elements for later use
    legend_handles = [el for el in legend_elements]
    legend_labels = [el.get_label() for el in legend_elements]

    return legend_handles, legend_labels


def lighten_color(color, amount=0.5):
    """
    Lighten the given color by blending it with white.
    
    Parameters:
    - color: RGB tuple or hex string of the color to lighten.
    - amount: Fraction by which to lighten the color (0 is no change, 1 is white).
    
    Returns:
    - A lighter RGB color.
    """
    try:
        c = mcolors.cnames[color]
    except:
        c = color
    c = mcolors.to_rgb(c)
    return [(1.0 - amount) * c[i] + amount for i in range(3)]


def save_figs_and_axes(figs_and_axes, output_dir, filename="saved_plots.pkl"):
    """Save the figure and axes data in a pickle file."""
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(figs_and_axes, f)
        

def load_figs_and_axes(output_dir, filename="saved_plots.pkl"):
    """Load the figure and axes data later."""
    with open(os.path.join(output_dir, filename), 'rb') as f:
        figs_and_axes = pickle.load(f)
    return figs_and_axes


def save_legend_image(legend_handles, legend_labels, output_dir, plot_type='absolute', filename="legend.png", font_size=12):
    fig, ax = plt.subplots(figsize=(1, 1))  # Dummy figure for legend
    ax.axis('off')

    # Legend is made out of 6 elements, the last 3 are lengthy in writing: use 3 columns (i.e., 2 rows)
    num_columns = 3
    
    # Save the legend
    legend = ax.legend(legend_handles, legend_labels, loc='center', ncol=num_columns, fontsize=font_size,
                       handler_map={patches.Patch: DiagonalColorHandler()} if plot_type == 'absolute' else None)
    fig.canvas.draw()
    
    # Save as image
    legend_filename = os.path.join(output_dir, filename)
    plt.savefig(legend_filename, dpi=OUT_DPI, bbox_inches='tight')
    plt.close()

    return legend_filename


## Section for further handling of the generated dataframes:

def create_dataframe(data, columns, model_name):
    # Convert the results dictionary to DataFrame
    results_df = pd.DataFrame.from_dict(data, orient='index')[columns]
    results_df.reset_index(inplace=True)  # Makes 'Metric' a column
    results_df.rename(columns={'index': 'Metric'}, inplace=True)  # Renames the new column to 'Metric'
    # Insert 'Model' column as the first column
    results_df.insert(0, 'Model', model_name)
    return results_df


def update_global_metric_ranges(combined_results_df, focus_metrics, models, combined_groups, global_metric_ranges):
    all_relevant_metrics = [metric for metric in combined_results_df['Metric'].unique()
                            if any(focus in metric for focus in focus_metrics)]
    
    # Initialise global metric ranges at the first call of this function
    new_metrics = [metric for metric in all_relevant_metrics if metric not in global_metric_ranges.keys()]
    for metric in new_metrics:
        global_metric_ranges[metric] = (float('inf'), -float('inf'))
    
    # Initialise local metric ranges, to be later compared to our glocal metric ranges
    local_metric_ranges = {metric: (float('inf'), -float('inf')) for metric in all_relevant_metrics}

    for model in models:
        for metric in all_relevant_metrics:
            data = combined_results_df[(combined_results_df['Model'] == model["fullname"]) & (combined_results_df['Metric'] == metric)]
            for group in combined_groups:
                # Fetch CI upper and lower values
                ci_data = data[group].apply(lambda x: x['ci']).iloc[0]
                ci_lower = ci_data[0]
                ci_upper = ci_data[1] 
                # Update metric range, including some margin
                local_metric_ranges[metric] = (min(local_metric_ranges[metric][0], ci_lower), 
                                               max(local_metric_ranges[metric][1], ci_upper))                

    # Add margins to the y-axis ranges and round values for cleaner axis limits
    for metric, (min_val, max_val) in local_metric_ranges.items():
        range_margin = (max_val - min_val) * 0.05  # 5% margin on each side
        adjusted_min = min_val - range_margin
        adjusted_max = max_val + range_margin
        # Dynamic rounding based on the order of magnitude of the range margin
        magnitude = -np.floor(np.log10(range_margin)) + 1
        rounding_factor = np.power(10, magnitude)
        floored_min = np.floor(adjusted_min * rounding_factor) / rounding_factor
        ceiled_max = np.ceil(adjusted_max * rounding_factor) / rounding_factor
        # Update the global metric ranges
        global_metric_ranges[metric] = (min(global_metric_ranges[metric][0], floored_min), 
                                        max(global_metric_ranges[metric][1], ceiled_max))


def read_csv_file(file_path):
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError(f"The file '{file_path}' is empty.")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        raise ValueError(f"The file '{file_path}' could not be parsed.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred while reading the file '{file_path}': {str(e)}")

# ========================================================
# ===== PERFORMANCE ANALYSIS - UTILS FUNCTIONS - END =====
# ========================================================



# ========================================================
# ======= BOOTSTRAPPING USING 2000 SAMPLES - START =======
# ========================================================

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

    finalise_metrics(metrics)
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
        metrics["TPR at global threshold"][group_name].append(tpr_global_thres)
        metrics["FPR at global threshold"][group_name].append(fpr_global_thres)
        metrics["Youden\'s Index at global threshold"][group_name].append(tpr_global_thres - fpr_global_thres)

        # Update aucroc_metrics for the first iteration only (no resampling), to be used for AUC-ROC plots
        if aucroc_metrics is not None:
            aucroc_metrics["AUC-ROC"][group_name] = auc_roc
            aucroc_metrics["TPRs"][group_name] = tpr
            aucroc_metrics["FPRs"][group_name] = fpr
            aucroc_metrics["TPR at threshold"][group_name] = tpr_global_thres
            aucroc_metrics["FPR at threshold"][group_name] = fpr_global_thres
            aucroc_metrics["Youden\'s Index at threshold"][group_name] = tpr_global_thres - fpr_global_thres


def calculate_subgroups_averages(metrics):
    subgroup_names = [group for group in metrics["AUC-ROC"].keys() if group != "All"]  # i.e., RACES + SEXES
    for metric_name in metrics.keys():
        for iteration in range(len(metrics[metric_name]["All"])):  # e.g., 'All' has entries for each bootstrap iteration
            # Compute the average value across the subgroups for this bootstrap iteration
            average = np.mean([metrics[metric_name][subgroup][iteration] for subgroup in subgroup_names])
            metrics[metric_name]["Average"].append(average)  # Effectively creates an 'Average' column (metrics is a defaultdict)


def calculate_relative_changes(metrics):
    # Copy the metric names to avoid modifying the dictionary while iterating
    absolute_metrics = list(metrics.keys())
    for metric_name in absolute_metrics:
        if "Average" in metrics[metric_name]:
            average_values = metrics[metric_name]["Average"]
            relative_metric_name = f"Relative {metric_name}"
            for subgroup in metrics[metric_name].keys():
                subgroup_values = metrics[metric_name][subgroup]
                relative_changes = [(subgroup_value - average) / average * 100 if average != 0 else 0
                                    for subgroup_value, average in zip(subgroup_values, average_values)]
                metrics[relative_metric_name][subgroup] = relative_changes


def finalise_metrics(metrics):
    """This function is to be called after the main bootstrap loop"""
    # Calculate averages for subgroups
    calculate_subgroups_averages(metrics)
    # Calculate relative changes compared to the subgroups averages
    calculate_relative_changes(metrics)


def aggregate_metrics_with_ci(metrics, ci_level=0.95, compact=False):
    summary = {}
    alpha = (1 - ci_level) / 2
    for metric, groups in metrics.items():
        summary[metric] = {}
        for group, values in groups.items():
            if len(values) > 1:  # Check that there's more than the initial value to calculate CI
                original_estimate = values[0]
                ci_lower = np.quantile(values[1:], alpha)
                ci_upper = np.quantile(values[1:], 1 - alpha)
                if compact:
                    summary[metric][group] = f"{original_estimate:.2f} ({ci_lower:.2f}-{ci_upper:.2f})"
                else:
                    summary[metric][group] = {
                        "estimate": original_estimate,
                        "ci": (ci_lower, ci_upper)
                    }
            else:  # Handling cases with insufficient data
                summary[metric][group] = {
                    "estimate": values[0],
                    "ci": (None, None)
                } if not compact else f"{values[0]:.2f} (N/A)"
    return summary

# ========================================================
# ======== BOOTSTRAPPING USING 2000 SAMPLES - END ========
# ========================================================



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ==== ABSOLUTE & RELATIVE PERFORMANCES PLOTS - START ====
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def plot_metrics(results_df, label, focus_metrics, y_limits, plot_type='absolute', models=None, RACES=None, SEXES=None, for_grid=False,
                 ci_capsize=4, ci_markersize=4, font_size=18, fig_size=(16,7), title_font_delta=4, axis_font_delta=1, line_width=0.8, 
                 y_xaxis_annotation=-28, title_pad = 12.5, label_pad=12.5, bars_cluster_width=0.7, output_dir=None, dataset_name='CheXpert'):
    """Generates and saves plots based on the provided data."""
    # Concatenate tab20b and tab20c color maps
    cmap_b = plt.get_cmap('tab20').colors
    cmap_c = plt.get_cmap('tab20c').colors
    full_cmap = list(cmap_b) + list(cmap_c)  # Concatenated colormap
    # Lighten every odd-numbered color in full_cmap
    for i in range(0, len(full_cmap), 2):
        full_cmap[i] = lighten_color(full_cmap[i], amount=0.2)  # Lighten by 20%

    # Determine which metrics and subgroups to plot based on plot_type
    if plot_type == 'absolute':
        metrics_to_plot = [metric for metric in results_df['Metric'].unique()
                           if any(focus in metric for focus in focus_metrics) and "Relative" not in metric]
        subgroups = RACES + SEXES + ['Average']
        all_group = ['All']  # Separate category for 'All'
    else:
        metrics_to_plot = [metric for metric in results_df['Metric'].unique()
                           if any(focus in metric for focus in focus_metrics) and "Relative" in metric]
        subgroups = RACES + SEXES  # No 'Average' or 'All'
        all_group = []
    combined_groups = subgroups + all_group

    # List to store figures and axes for later use
    figs_and_axes = []

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=fig_size)
        plot_bars(ax, results_df, metric, models, combined_groups, full_cmap, bars_cluster_width, line_width, ci_capsize, ci_markersize, font_size)

        # Adjust plot aesthetics
        ax.set_ylim(*y_limits[metric])
        set_plot_labels(ax, label, metric, plot_type, font_size, title_pad, label_pad, title_font_delta, axis_font_delta, dataset_name)
        legend_handles, legend_labels = create_custom_legend(ax, models, full_cmap, plot_type, BAR_EDGE_COLOUR, ERROR_EDGE_COLOUR, 
                                                             font_size, line_width)

        # Annotation for Subgroups
        midpoint = len(subgroups) / 2 - 0.5 
        ax.annotate('Subgroups', xy=(midpoint, ax.get_ylim()[0]), xycoords='data',
            xytext=(0, y_xaxis_annotation), textcoords='offset points', ha='center', va='top', fontsize=font_size+axis_font_delta)
        
        # Annotation for All group
        if "All" in combined_groups:
            width = bars_cluster_width / len(models)
            ax.annotate('All', xy=(len(subgroups) + width, ax.get_ylim()[0]), xycoords='data',
                xytext=(0, y_xaxis_annotation), textcoords='offset points', ha='center', va='top', fontsize=font_size+axis_font_delta)

        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray', alpha=0.5, zorder=1)

        # Save individual plot
        if for_grid:
            plot_filename = f"{plot_type}_gridready__{metric.replace(' ', '_').lower()}__({dataset_name}--{label.replace(' ', '_')}).png"
            ax.legend_.remove()
            ax.set_title(f"{label}", fontsize=font_size+title_font_delta)
        else:
            plot_filename = f"{plot_type}__{metric.replace(' ', '_').lower()}__({dataset_name}--{label.replace(' ', '_')}).png"
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, plot_filename), dpi=OUT_DPI)
        plt.close()

        figs_and_axes.append((fig, ax))

    save_figs_and_axes(figs_and_axes, output_dir, filename=f"{plot_type}_figs_and_axes__({dataset_name}--{label.replace(' ', '_')})")

    # Return the list of figures and axes, and the shared legend
    return metrics_to_plot, legend_handles, legend_labels


def plot_bars(ax, results_df, metric, models, groups, full_cmap, bars_cluster_width, line_width, ci_capsize, ci_markersize, font_size):
    # Grouping bars for each model within each subgroup
    width = bars_cluster_width / len(models)  # width of each bar in group of bars, distributed among models
    margin = 1*width
    xticks_positions = np.linspace(0, len(groups)-1, len(groups))
    if "All" in groups:
        xticks_positions[-1] += margin
    
    for i, model in enumerate(models):
        # Filter data for each model
        model_data = results_df[(results_df['Model'] == model["fullname"]) & (results_df['Metric'] == metric)]
        # Initialise lists to store the extracted values and confidence intervals
        values = []
        ci_errors = []
        
        # Extract estimate and confidence intervals for each subgroup
        for group in groups:
            group_data = model_data.iloc[0][group]  # Assuming single row per model and metric
            values.append(group_data['estimate'])
            ci_lower, ci_upper = group_data['ci']
            ci_errors.append([group_data['estimate'] - ci_lower, ci_upper - group_data['estimate']])

        # Convert lists to numpy arrays for plotting
        values = np.array(values)
        ci_errors = np.array(ci_errors).T  # Transpose to fit expected shape for error bars

        # Define colour scheme per model based on concatenated colormap
        model_colours = full_cmap[i*4:(i+1)*4]  # Extract 4 colors for each model
        # Correct colour order for subgroups
        model_colours = [model_colours[3]] * len(RACES) + [model_colours[3]] * len(SEXES) + [model_colours[2]] + [model_colours[2]]
        
        # Set position for each bar
        positions = xticks_positions + i * width - (width * (len(models) - 1) / 2)
        # Plot each bar
        ax.bar(positions, values, width=width, label=model["shortname"], color=model_colours, edgecolor=BAR_EDGE_COLOUR, linewidth=line_width, zorder=3)

        # Plot error bars separately
        for pos, val, err in zip(positions, values, ci_errors.T):
            ax.errorbar(pos, val, yerr=[[err[0]], [err[1]]], fmt='none', ecolor=ERROR_EDGE_COLOUR, capsize=ci_capsize, elinewidth=line_width, zorder=3)
            ax.plot(pos, val, 'o', color=ERROR_EDGE_COLOUR, markersize=ci_markersize, zorder=4)

    # Add vertical dotted line if 'All' is part of the groups
    if 'All' in groups:
        line_position = len(groups) - 1.5 + margin/2
        ax.axvline(x=line_position, color='black', linestyle='dashed', dashes=(6, 8), linewidth=line_width, zorder=2)

    # Formatting the plot
    ax.set_xticks(xticks_positions)
    ax.set_xticklabels(groups[:-1]+[""], fontsize=font_size) if 'All' in groups else ax.set_xticklabels(groups, fontsize=font_size)


def aggregate_plots_into_grid(output_dir, metrics_to_plot, labels, plot_type, grid_shape=(2, 2), figsize=(16, 8), font_size=12, 
                              title_font_delta=4, legend_image_path=None, legend_height_fraction=0.15, wspace=-0.15, hspace=0.05, 
                              dataset_name='CheXpert'):
    """Aggregates the figures into a grid and displays them."""
    for i, metric in enumerate(metrics_to_plot):
        if legend_image_path:
            nrows = grid_shape[0] + 1  # Adding an extra row for the legend
            height_ratios = [1] * grid_shape[0] + [legend_height_fraction]
        else:
            nrows = grid_shape[0]
            height_ratios = [1] * grid_shape[0]

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows=nrows, 
                               ncols=grid_shape[1], 
                               width_ratios=[1] * grid_shape[1], 
                               height_ratios=height_ratios)

        axes = [fig.add_subplot(gs[row, col]) for row in range(grid_shape[0]) for col in range(grid_shape[1])]

        # Add title
        title = get_plot_title(metric, plot_type, dataset_name)
        fig.suptitle(title, fontsize=font_size+title_font_delta)

        for j, label in enumerate(labels):
            # Calculate the row and column of the subplot in the grid based on the index `j`
            ax_subplot = axes[j]

            # Construct the filename for the grid-ready plot
            plot_filename_grid = f"{plot_type}_gridready__{metric.replace(' ', '_').lower()}__({dataset_name}--{label.replace(' ', '_')}).png"
            plot_filepath = os.path.join(output_dir, plot_filename_grid)

            # Load the image and display it in the subplot
            img = mpimg.imread(plot_filepath)
            ax_subplot.imshow(img)
            ax_subplot.axis('off')
 
        # Load and add the legend as an image (if available)
        if legend_image_path:
            ax_legend = fig.add_subplot(gs[-1, :])  # Last row, spanning all columns
            ax_legend.axis('off')
            img_legend = mpimg.imread(legend_image_path)
            ax_legend.imshow(img_legend)
        
        grid_plot_filename = f"{plot_type}_grid_combined__{metric.replace(' ', '_').lower()}__({dataset_name}).png"
        plt.tight_layout()
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(os.path.join(output_dir, grid_plot_filename), dpi=OUT_DPI)
        plt.close()

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ===== ABSOLUTE & RELATIVE PERFORMANCES PLOTS - END =====
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# =============== ROC CURVE PLOTS - START ================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def plot_auc_roc_curves(aucroc_metrics_df, label, output_dir, subgroups, models=None, dataset_name='CheXpert', lw=1.5, alpha=0.8, 
                        markersize=12, fig_size=(8, 5), font_size=12, title_font_delta=2, axis_font_delta=1, legend_font_delta=-1, 
                        for_grid=False):
    for model in models:
        model_data = aucroc_metrics_df[(aucroc_metrics_df['Model'] == model['fullname'])]
        _, ax = plt.subplots(figsize=fig_size)
        original_cmap = plt.get_cmap('Dark2')
        colors = list(original_cmap(np.linspace(0, 1, len(subgroups))))
        colors[2] = 'mediumvioletred'  # Replace third colour

        for idx, subgroup in enumerate(subgroups):
            fprs = model_data[model_data['Metric'] == 'FPRs'][subgroup].iloc[0]
            tprs = model_data[model_data['Metric'] == 'TPRs'][subgroup].iloc[0]
            auc_score = model_data[model_data['Metric'] == 'AUC-ROC'][subgroup].iloc[0]
            plt.plot(fprs, tprs, lw=lw, alpha=alpha, label=f'{subgroup} AUC-ROC={auc_score:.2f}', color=colors[idx])

        plt.gca().set_prop_cycle(None)
            
        for idx, subgroup in enumerate(subgroups):
            # Plotting the global threshold point
            tpr_global_thresh = model_data[model_data['Metric'] == 'TPR at threshold'][subgroup].iloc[0]
            fpr_global_thresh = model_data[model_data['Metric'] == 'FPR at threshold'][subgroup].iloc[0]
            plt.plot(fpr_global_thresh, tpr_global_thresh, marker='*', linestyle='None', alpha=alpha, markersize=markersize,
                        label=f'{subgroup} TPR={tpr_global_thresh:.2f}, FPR={fpr_global_thresh:.2f}', color=colors[idx])

        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='midnightblue', alpha=alpha)
        plt.annotate('Random Classifier', xy=(0.5, 0.52), fontsize=font_size-0.5, color='midnightblue', rotation=32)
        
        plt.xlabel('False Positive Rate (FPR)', fontsize=font_size + axis_font_delta)
        plt.ylabel('True Positive Rate (TPR)', fontsize=font_size + axis_font_delta)

        model_shortname = model['shortname'].replace(' ', '_')
        model_aucroc_dir_path = os.path.join(output_dir, model_shortname)
        os.makedirs(model_aucroc_dir_path, exist_ok=True)

        if for_grid:
            title = f"{label}"
            filename_suffix = "roc_curve_gridready"
        else:
            title = f"{model['shortname']}\nROC Curve — {label} | {dataset_name}"
            filename_suffix = "roc_curve"

        plt.title(title, fontsize=font_size + title_font_delta)
        ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
        plt.legend(loc='lower right', fontsize=font_size + legend_font_delta, ncol=2)
        ax.spines[['right', 'top']].set_visible(False)
        plt.grid(True, linestyle='-', linewidth=0.5, color='lightgray')
        
        filepath = os.path.join(model_aucroc_dir_path, f'{model_shortname}__{filename_suffix}__({dataset_name}--{label.replace(" ", "_")}).png')
        plt.savefig(filepath, dpi=OUT_DPI)
        plt.close()


def aggregate_roccurves_into_grid(output_dir, models, labels, grid_shape=(2, 2), figsize=(16, 8), font_size=12, 
                                  title_font_delta=4, wspace=-0.15, hspace=0.05, dataset_name='CheXpert'):
    """Aggregates the roc curves into a grid and displays them."""
    for i, model in enumerate(models):
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(nrows=grid_shape[0], 
                               ncols=grid_shape[1], 
                               width_ratios=[1] * grid_shape[1], 
                               height_ratios=[1] * grid_shape[0])

        axes = [fig.add_subplot(gs[row, col]) for row in range(grid_shape[0]) for col in range(grid_shape[1])]

        model_shortname = model['shortname'].replace(' ', '_')
        model_aucroc_dir_path = os.path.join(output_dir, model_shortname)

        # Add title
        title = f"{model['shortname']}\nROC Curve | {dataset_name}"
        fig.suptitle(title, fontsize=font_size+title_font_delta)

        for j, label in enumerate(labels):
            # Calculate the row and column of the subplot in the grid based on the index `j`
            ax_subplot = axes[j]

            # Construct the filename for the grid-ready plot
            roccurve_filename_grid = f'{model_shortname}__roc_curve_gridready__({dataset_name}--{label.replace(" ", "_")}).png'
            roccurve_filepath = os.path.join(model_aucroc_dir_path, roccurve_filename_grid)

            # Load the image and display it in the subplot
            img = mpimg.imread(roccurve_filepath)
            ax_subplot.imshow(img)
            ax_subplot.axis('off')
        
        grid_plot_filename = f'{model_shortname}__roc_curves_grid_combined__({dataset_name}--{label.replace(" ", "_")}).png'
        plt.tight_layout()
        plt.subplots_adjust(wspace=wspace, hspace=hspace)
        plt.savefig(os.path.join(output_dir, grid_plot_filename), dpi=OUT_DPI)
        plt.close()

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ================ ROC CURVE PLOTS - END =================
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



def parse_args():
    parser = argparse.ArgumentParser(description="Calculate performance metrics for classification model.")
    parser.add_argument('--models_dir', nargs='+', required=True, help='List ot paths to outputs directory from the different mdels we want to compare')
    parser.add_argument('--models_shortnames', nargs='+', required=True, help='List of short names of the models corresponding to the directories')
    parser.add_argument('--config', default='chexpert', choices=['chexpert', 'mimic'], help='Config dataset module to use')
    parser.add_argument('--labels', nargs='+', default=["Pleural Effusion", "No Finding", "Cardiomegaly", "Pneumothorax"], help='List of labels to process')
    parser.add_argument('--focus_metrics', default=["Youden\'s Index", "AUC-ROC"], help='Performance metrics to focus on for comparative analysis')
    parser.add_argument('--local_execution', type=bool, default=True, help='Boolean to check whether the code is run locally or remotely')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    # Load the configuration dynamically based on the command line argument
    config = load_config(args.config)
    # Accessing the configuration to import dataset-specific variables
    dataset_name = get_dataset_name(args.config)

    if args.local_execution:
        # Construct local file path dynamically based on the dataset name
        local_base_path = '/Users/macuser/Desktop/Imperial/70078_MSc_AI_Individual_Project/code/Test_Resample_Records'
        local_dataset_path = os.path.join(local_base_path, dataset_name)
        local_filename = os.path.basename(config.TEST_RECORDS_CSV)  # Extracts filename from the config path
        TEST_RECORDS_CSV = os.path.join(local_dataset_path, local_filename)
    else:
        TEST_RECORDS_CSV = config.TEST_RECORDS_CSV

    # Path to base output directory and data characteristics files
    base_output_path = os.getcwd()  # Use the current working directory for outputs
    data_characteristics = pd.read_csv(TEST_RECORDS_CSV)

    # Evaluation parameters
    target_fpr = TARGET_FPR
    n_bootstrap = N_BOOTSTRAP
    ci_level = CI_LEVEL

    # Directories to save analysis outputs
    performance_for_bias_dir_path = os.path.join(base_output_path, f'performance_for_bias--{dataset_name}')
    performance_tables_dir_path = os.path.join(performance_for_bias_dir_path, 'performance_tables/')
    absolute_plots_dir_path = os.path.join(performance_for_bias_dir_path, 'performance_plots_absolute/')
    relative_plots_dir_path = os.path.join(performance_for_bias_dir_path, 'performance_plots_relative/')
    aucroc_plots_dir_path = os.path.join(performance_for_bias_dir_path, 'aucroc_plots/')
    aucroc_tables_dir_path = os.path.join(performance_for_bias_dir_path, 'aucroc_tables/')
    os.makedirs(performance_for_bias_dir_path, exist_ok=True)
    os.makedirs(performance_tables_dir_path, exist_ok=True)
    os.makedirs(absolute_plots_dir_path, exist_ok=True)
    os.makedirs(relative_plots_dir_path, exist_ok=True)
    os.makedirs(aucroc_plots_dir_path, exist_ok=True)
    os.makedirs(aucroc_tables_dir_path, exist_ok=True)


    # Focus is on Youden's Index and AUC-ROC metrics
    focus_metrics = args.focus_metrics
    # Define all groups to be analysed
    all_combined_groups = RACES + SEXES + ["Average", "All"]
    all_combined_groups_noavg = RACES + SEXES + ["All"]

    # Initial empty dictionaries for storing combined dataframes and global metric ranges
    all_combined_results = {}
    all_combined_aucroc = {}
    all_combined_results_with_ci = {}
    all_combined_results_with_ci_compact = {}
    global_metric_ranges = {}

    # List to store models info
    models = []
    for model_dir, model_shortname in zip(args.models_dir, args.models_shortnames):
        model_fullname = os.path.basename(model_dir)
        models.append({
            "directory": model_dir,
            "fullname" : model_fullname,
            "shortname": model_shortname
        })
    

    # Iterate over each label of interest
    for label in args.labels:
        all_results_df = []
        all_aucroc_metrics_df = []
        all_results_with_ci_df = []
        all_results_with_ci_compact_df = []

        # For each label, iterate over each of the models for an aggregate comparison
        for model in models:
            outputs_csv_filepath = os.path.join(model["directory"], 'outputs_test.csv')
            model_outputs = read_csv_file(outputs_csv_filepath)

            label_index = LABELS.index(label)
            probs = model_outputs[f"prob_class_{label_index+1}"]
            targets = np.array(model_outputs[f"target_class_{label_index+1}"])
            races = data_characteristics.race.values
            sexes = data_characteristics.sex.values

            # Calculate metrics
            results, aucroc_metrics = bootstrap_ci(targets=targets, probs=probs, races=races, sexes=sexes, 
                                                n_bootstrap=n_bootstrap, target_fpr=target_fpr)
            results_with_ci = aggregate_metrics_with_ci(results, ci_level, compact=False)
            results_with_ci_compact = aggregate_metrics_with_ci(results, ci_level, compact=True)
            
            # Convert the result dictionnaries to DataFrame and add 'Model' column
            results_df = create_dataframe(data=results, columns=all_combined_groups, model_name=model["fullname"])
            aucroc_metrics_df = create_dataframe(data=aucroc_metrics, columns=all_combined_groups_noavg, model_name=model["fullname"])
            results_with_ci_df = create_dataframe(data=results_with_ci, columns=all_combined_groups, model_name=model["fullname"])
            results_with_ci_compact_df = create_dataframe(data=results_with_ci_compact, columns=all_combined_groups, model_name=model["fullname"])

            # Append dataframes to later ocncatenate them for all models
            all_results_df.append(results_df)
            all_aucroc_metrics_df.append(aucroc_metrics_df)
            all_results_with_ci_df.append(results_with_ci_df)
            all_results_with_ci_compact_df.append(results_with_ci_compact_df)

        # Combine dataframes for all models
        combined_results_df = pd.concat(all_results_df)
        combined_aucroc_metrics_df = pd.concat(all_aucroc_metrics_df)
        combined_results_with_ci_df = pd.concat(all_results_with_ci_df)
        combined_results_with_ci_compact_df = pd.concat(all_results_with_ci_compact_df)

        # Store combined dataframes in dedicated dictionaries
        all_combined_results[label] = combined_results_df
        all_combined_aucroc[label] = combined_aucroc_metrics_df
        all_combined_results_with_ci[label] = combined_results_with_ci_df
        all_combined_results_with_ci_compact[label] = combined_results_with_ci_compact_df

        # Store Performance Table:
        combined_results_df.to_csv(os.path.join(performance_tables_dir_path, f'all_performance_metrics__({dataset_name}--{label.replace(" ", "_")}).csv'), index=True)
        combined_results_with_ci_df.to_csv(os.path.join(performance_tables_dir_path, f'ci_performance_metrics--full__({dataset_name}--{label.replace(" ", "_")}).csv'), index=True)
        combined_results_with_ci_compact_df.to_csv(os.path.join(performance_tables_dir_path, f'ci_performance_metrics--compact__({dataset_name}--{label.replace(" ", "_")}).csv'), index=True)
        print(f"\nResults for: {label.upper()} ({ci_level * 100:.0f}%-CI with {n_bootstrap} bootstrap samples)")
        print(tabulate(combined_results_with_ci_compact_df, headers=combined_results_with_ci_compact_df.columns))

        # Store AUC-ROC Table:
        combined_aucroc_metrics_df.to_csv(os.path.join(aucroc_tables_dir_path, f'all_aucroc_metrics__({dataset_name}--{label.replace(" ", "_")}).csv'), index=True)

        # Update global metric ranges:
        update_global_metric_ranges(combined_results_df=combined_results_with_ci_df, focus_metrics=focus_metrics, models=models,
                                    combined_groups=all_combined_groups, global_metric_ranges=global_metric_ranges)
                

    # >>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Generate plots for each label (for all metrics in focus_metrics): Plot for individual use
    for label in args.labels:
        # Plot Absolute Performance
        _, _, _ = plot_metrics(
            results_df=all_combined_results_with_ci[label], label=label, focus_metrics=focus_metrics, y_limits=global_metric_ranges, 
            plot_type='absolute', models=models, RACES=RACES, SEXES=SEXES, for_grid=False, ci_capsize=4, ci_markersize=4, font_size=18, 
            fig_size=(16,7), title_font_delta=4, axis_font_delta=1, line_width=0.8, y_xaxis_annotation=-28, title_pad = 12.5, 
            label_pad=12.5, bars_cluster_width=0.7, output_dir=absolute_plots_dir_path, dataset_name=dataset_name
            )
        # Plot Relative Performance
        _, _, _ = plot_metrics(
            results_df=all_combined_results_with_ci[label], label=label, focus_metrics=focus_metrics, y_limits=global_metric_ranges, 
            plot_type='relative', models=models, RACES=RACES, SEXES=SEXES, for_grid=False, ci_capsize=4, ci_markersize=4, font_size=18, 
            fig_size=(16,7), title_font_delta=4, axis_font_delta=1, line_width=0.8, y_xaxis_annotation=-28, title_pad = 12.5, 
            label_pad=12.5, bars_cluster_width=0.7, output_dir=relative_plots_dir_path, dataset_name=dataset_name
            )
        # Plot AUC-ROC curves
        plot_auc_roc_curves(aucroc_metrics_df=all_combined_aucroc[label], label=label, output_dir=aucroc_plots_dir_path, 
                            subgroups=all_combined_groups_noavg, models=models, dataset_name=dataset_name, lw=1.5, alpha=0.8, 
                            markersize=12, fig_size=(8, 5), font_size=12, title_font_delta=2, axis_font_delta=1, 
                            legend_font_delta=-1, for_grid=False)


    # >>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Generate plots for each label (for all metrics in focus_metrics): Plot for 2x2 grid use
    font_size_plotforgrid = 20
    fig_size_plotforgrid = (10,6)
    title_font_delta = 4
    font_size_roccurveforgrid = 16
    fig_size_roccurveforgrid = (8,5)
    for label in args.labels:
        # Plot Absolute Performance
        metrics_to_plot_abs, legend_handles_abs, legend_labels_abs = plot_metrics(
            results_df=all_combined_results_with_ci[label], label=label, focus_metrics=focus_metrics, y_limits=global_metric_ranges, 
            plot_type='absolute', models=models, RACES=RACES, SEXES=SEXES, for_grid=True, ci_capsize=3, ci_markersize=4, 
            font_size=font_size_plotforgrid, fig_size=fig_size_plotforgrid, title_font_delta=title_font_delta, axis_font_delta=1.5, 
            line_width=0.8, y_xaxis_annotation=-28, title_pad = 20, label_pad=12.5, bars_cluster_width=0.7, 
            output_dir=absolute_plots_dir_path, dataset_name=dataset_name
            )
        # Plot Relative Performance
        metrics_to_plot_rel, legend_handles_rel, legend_labels_rel = plot_metrics(
            results_df=all_combined_results_with_ci[label], label=label, focus_metrics=focus_metrics, y_limits=global_metric_ranges, 
            plot_type='relative', models=models, RACES=RACES, SEXES=SEXES, for_grid=True, ci_capsize=3, ci_markersize=4, 
            font_size=font_size_plotforgrid, fig_size=fig_size_plotforgrid, title_font_delta=title_font_delta, axis_font_delta=1.5, 
            line_width=0.8, y_xaxis_annotation=-28, title_pad = 20, label_pad=12.5, bars_cluster_width=0.7, 
            output_dir=relative_plots_dir_path, dataset_name=dataset_name
            )
        # Plot AUC-ROC curves
        plot_auc_roc_curves(aucroc_metrics_df=all_combined_aucroc[label], label=label, output_dir=aucroc_plots_dir_path, 
                            subgroups=all_combined_groups_noavg, models=models, dataset_name=dataset_name, lw=1.5, alpha=0.8, 
                            markersize=12, fig_size=fig_size_roccurveforgrid, font_size=font_size_roccurveforgrid, 
                            title_font_delta=2, axis_font_delta=1, legend_font_delta=-4, for_grid=True)
        

    # >>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<
    # Save the legend as an image
    legend_image_path_abs = save_legend_image(legend_handles=legend_handles_abs, legend_labels=legend_labels_abs, output_dir=absolute_plots_dir_path, 
                                        plot_type='absolute', filename=f"legend_abs__({dataset_name}).png", font_size=font_size_plotforgrid)
    legend_image_path_rel = save_legend_image(legend_handles=legend_handles_rel, legend_labels=legend_labels_rel, output_dir=relative_plots_dir_path, 
                                        plot_type='relative', filename=f"legend_rel__({dataset_name}).png", font_size=font_size_plotforgrid)
    
    # Combine related plots (come by 4) into a single 2x2 grid
    # Absolute Performance Grid
    aggregate_plots_into_grid(
        output_dir=absolute_plots_dir_path, metrics_to_plot=metrics_to_plot_abs, labels=args.labels, plot_type='absolute', grid_shape=(2, 2), 
        font_size=font_size_plotforgrid, figsize=(fig_size_plotforgrid[0]*2, fig_size_plotforgrid[1]*2), title_font_delta=title_font_delta+2, 
        legend_image_path=legend_image_path_abs, legend_height_fraction=0.2, wspace=-0.15, hspace=0, dataset_name=dataset_name
        )
    # Relative Performance Grid
    aggregate_plots_into_grid(
        output_dir=relative_plots_dir_path, metrics_to_plot=metrics_to_plot_rel, labels=args.labels, plot_type='relative', grid_shape=(2, 2), 
        font_size=font_size_plotforgrid, figsize=(fig_size_plotforgrid[0]*2, fig_size_plotforgrid[1]*2), title_font_delta=title_font_delta+2, 
        legend_image_path=legend_image_path_rel, legend_height_fraction=0.2, wspace=-0.15, hspace=0, dataset_name=dataset_name
        )
    # ROC Curves Grid
    aggregate_roccurves_into_grid(
        output_dir=aucroc_plots_dir_path, models=models, labels=args.labels, grid_shape=(2, 2), title_font_delta=4, wspace=-0.2, hspace=0,
        figsize=(fig_size_roccurveforgrid[0]*2, fig_size_roccurveforgrid[1]*2), font_size=font_size_roccurveforgrid, dataset_name=dataset_name
        )
 
