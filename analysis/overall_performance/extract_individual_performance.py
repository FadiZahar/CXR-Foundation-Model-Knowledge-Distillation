import os
import argparse
import numpy as np
import pandas as pd

# Import global shared variables
from config.config_shared import TARGET_FPR, LABELS, LABELS_BY_RELEVANCE
# Import the configuration loader
from config.loader_config import load_config, get_dataset_name

# Import performace metrics calculation functions
from utils.output_utils.generate_and_save_metrics import calculate_roc_auc, calculate_pr_auc, calculate_youden_index
# Import other utils functions
from analysis.bias_analysis.evaluate_disease_prediction import read_csv_file


# Defining global variables
MAX_EPOCH_NUMBER = 19  # 20 epochs in total, starting from 0 to 19
METRICS = [
    "AUC-ROC", 
    "AUC-PR", 
    "Maximum Youden\'s J Statistic", 
    f"Youden\'s J Statistic at {int(TARGET_FPR*100)}% FPR"
    ]
METRICS_EPOCH_ACTIONS = ["Epoch at Lowest", "Epoch at Peak", "Epoch at Convergence"]
CLASSES = [f"Class {idx+1} [{LABELS_BY_RELEVANCE[idx]}]" for idx in range(len(LABELS_BY_RELEVANCE))]

CUSTOM_TRACKING_ORDER_DETAILED = [
    ('Validation Loss', 'All Classes', action) for action in METRICS_EPOCH_ACTIONS
] + [
    (metric, class_label, action)
    for metric in METRICS
    for class_label in CLASSES + ["All Classes"]
    for action in METRICS_EPOCH_ACTIONS
    if not (metric == 'Validation Loss' and class_label == 'All Classes')
]
AVERAGE_FOCUS_CLASS_NAME = "Average Classes 1 to 7"
OTHERS_CLASS_NAME = "Others"

np.random.seed(42)



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ======== PERFORMANCE METRICS COMPUTATION - START =======
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def get_custom_tracking_order_focused(focus_classes, average_focus_class_name, others_class_name):
    CUSTOM_TRACKING_ORDER_FOCUSED = [
        ('Validation Loss', 'All Classes', action) for action in METRICS_EPOCH_ACTIONS
    ] + [
        (metric, class_label, action)
        for metric in METRICS
        for class_label in focus_classes + [average_focus_class_name, others_class_name]
        for action in METRICS_EPOCH_ACTIONS
        if not (metric == 'Validation Loss' and class_label == 'All Classes')
    ]
    return CUSTOM_TRACKING_ORDER_FOCUSED


def modify_results_for_focus_labels(results_df, focus_classes, other_classes, metrics, models_names, 
                                    average_focus_class_name, others_class_name):
    # Create a copy to preserve the original results
    focused_results_df = results_df.copy()
    
    # Adding an 'Average Focus Classes (i.e., Classes 1 to 7)' and 'Others' row
    for metric in metrics:
        for model_name in models_names:
            focus_values = results_df.loc[(metric, focus_classes), model_name].astype(float)
            other_values = results_df.loc[(metric, other_classes), model_name].astype(float)
            # Insert the average for focus classes
            focused_results_df.loc[(metric, average_focus_class_name), model_name] = focus_values.mean()
            # Insert the average for other classes
            focused_results_df.loc[(metric, others_class_name), model_name] = other_values.mean()

    # Recompute average and standard deviation for focus labels and 'Others'
    for metric in metrics:
        for class_label in [average_focus_class_name, others_class_name]:
            values = focused_results_df.loc[(metric, class_label), models_names].astype(float)
            avg = values.mean()
            stddev = values.std()
            percent_stddev = (stddev/avg)*100 if avg != 0 else 0
            focused_results_df.loc[(metric, class_label), 'Average'] = avg
            focused_results_df.loc[(metric, class_label), 'SD'] = stddev
            focused_results_df.loc[(metric, class_label), '%SD'] = percent_stddev
            focused_results_df.at[(metric, class_label), 'AverageSD Detailed'] = {'Mean': avg, 'SD': stddev, '%SD': percent_stddev}
            focused_results_df.loc[(metric, class_label), 'AverageSD Compact'] = f"{avg:.2f} ± {stddev:.2f} (± {percent_stddev:.2f}%)"

    # Resort the DataFrame to order the classes per metric correctly
    class_order = focus_classes + [average_focus_class_name, others_class_name]
    focused_results_df = focused_results_df.reindex(pd.MultiIndex.from_product([metrics, class_order], names=['Metric', 'Class']))

    # Final structuring: resetting the index of the DataFrame to convert the multi-level index into columns.
    focused_results_df.reset_index(inplace=True)
    focused_results_df.rename(columns={'level_0': 'Metric', 'level_1': 'Class'}, inplace=True)
    
    return focused_results_df


def modify_tracking_metrics_for_focus_labels(tracking_df, focus_classes, other_classes, metrics, models_names, metrics_actions, 
                                             custom_tracking_order_focused, average_focus_class_name, others_class_name):
    # Create a copy to preserve the original results
    focused_tracking_df = tracking_df.copy()
    
    # Adding an 'Average Focus Classes (i.e., Classes 1 to 7)' and 'Others' row
    for metric in metrics:
        for metric_action in metrics_actions:
            for model_name in models_names:
                focus_values = tracking_df.loc[(metric, focus_classes, metric_action), model_name].astype(float)
                other_values = tracking_df.loc[(metric, other_classes, metric_action), model_name].astype(float)
                # Insert the average for focus classes
                focused_tracking_df.loc[(metric, average_focus_class_name, metric_action), model_name] = focus_values.mean()
                # Insert the average for other classes
                focused_tracking_df.loc[(metric, others_class_name, metric_action), model_name] = other_values.mean()

    # Recompute average and standard deviation for focus labels and 'Others'
    for metric in metrics:
        for class_label in [average_focus_class_name, others_class_name]:
            for metric_action in metrics_actions:
                values = focused_tracking_df.loc[(metric, class_label, metric_action), models_names].astype(float)
                avg = values.mean()
                stddev = values.std()
                percent_stddev = (stddev/avg)*100 if avg != 0 else 0
                focused_tracking_df.loc[(metric, class_label, metric_action), 'Average'] = avg
                focused_tracking_df.loc[(metric, class_label, metric_action), 'SD'] = stddev
                focused_tracking_df.loc[(metric, class_label, metric_action), '%SD'] = percent_stddev
                focused_tracking_df.at[(metric, class_label, metric_action), 'AverageSD Detailed'] = {'Mean': avg, 'SD': stddev, '%SD': percent_stddev}
                focused_tracking_df.loc[(metric, class_label, metric_action), 'AverageSD Compact'] = f"{avg:.2f} ± {stddev:.2f} (± {percent_stddev:.2f}%)"
    
    # Final structuring: resetting the index of the DataFrame to convert the multi-level index into columns.
    focused_tracking_df = focused_tracking_df.reindex(pd.MultiIndex.from_tuples(custom_tracking_order_focused, names=['Metric', 'Class', 'Metric Action']))
    focused_tracking_df.reset_index(inplace=True)
    focused_tracking_df.rename(columns={'level_0': 'Metric', 'level_1': 'Class', 'level_2': 'Metric Action'}, inplace=True)
    
    return focused_tracking_df


def extract_epoch_from_checkpoint(dir_path):
    # Assuming there is only one checkpoint file per model directory
    for file in os.listdir(dir_path):
        if file.startswith("best-checkpoint") and file.endswith(".ckpt"):
            epoch_num = int(file.split('epoch=')[1].split('-')[0])
            return epoch_num
    return None


def epoch_at_lowest(metric_values):
    """ Returns the epoch number where the metric has its minimum value. """
    return metric_values.index(min(metric_values))

def epoch_at_peak(metric_values):
    """ Returns the epoch number where the metric has its maximum value. """
    return metric_values.index(max(metric_values))

def epoch_at_convergence(metric_values, window=3, threshold_ratio=0.03):
    """
    Returns the epoch number where the metric stabilises based on a moving average.
    Convergence is defined when changes between the moving averages of subsequent epochs
    fall below a dynamic threshold set as a ratio of the metric's value range.
    """
    if len(metric_values) < window + 1:
        return None  # Not enough data to compute the desired moving average
    
    # Calculate the moving average
    moving_averages = [sum(metric_values[i:i+window]) / window for i in range(len(metric_values) - window + 1)]
    
    # Calculate the dynamic threshold
    value_range = max(metric_values) - min(metric_values)
    dynamic_threshold = value_range * threshold_ratio
    
    # Check for convergence in moving averages
    for i in range(1, len(moving_averages)):
        if abs(moving_averages[i] - moving_averages[i - 1]) < dynamic_threshold:
            return i - 1  # Return the epoch number adjusted for the window

    return None


def populate_tracking_df(metric_values, metric, class_label, model_name, tracking_df, all_classes_epoch_results=None, 
                         epoch_from_checkpoint=None,):
    # Calculate epoch metrics
    epoch_lowest = epoch_at_lowest(metric_values)
    epoch_peak = epoch_at_peak(metric_values)
    epoch_converged = epoch_at_convergence(metric_values)

    # Compare with extracted epoch from checkpoint if metric is val_loss
    if epoch_from_checkpoint and epoch_lowest != epoch_from_checkpoint:
        epoch_lowest = f"{epoch_lowest} - {epoch_from_checkpoint}*"

    # Populate tracking_df for the given metric and class
    tracking_df.loc[(metric, class_label, 'Epoch at Lowest'), model_name] = epoch_lowest
    tracking_df.loc[(metric, class_label, 'Epoch at Peak'), model_name] = epoch_peak
    tracking_df.loc[(metric, class_label, 'Epoch at Convergence'), model_name] = epoch_converged

    # Append epoch results to the global metrics collection
    if all_classes_epoch_results and class_label != "All Classes":  
        all_classes_epoch_results[metric]['Epoch at Lowest'].append(epoch_lowest)
        all_classes_epoch_results[metric]['Epoch at Peak'].append(epoch_peak)
        all_classes_epoch_results[metric]['Epoch at Convergence'].append(epoch_converged)
    # Averaging for global epoch results, for "All Classes"
    elif all_classes_epoch_results and class_label == "All Classes":  
        # Filter None values before averaging
        valid_epochs_lowest = [e for e in all_classes_epoch_results[metric]['Epoch at Lowest'] if e is not None]
        valid_epochs_peak = [e for e in all_classes_epoch_results[metric]['Epoch at Peak'] if e is not None]
        valid_epochs_convergence = [e for e in all_classes_epoch_results[metric]['Epoch at Convergence'] if e is not None]
        tracking_df.loc[(metric, class_label, 'Epoch at Lowest'), model_name] = np.mean(valid_epochs_lowest) if valid_epochs_lowest else None
        tracking_df.loc[(metric, class_label, 'Epoch at Peak'), model_name] = np.mean(valid_epochs_peak) if valid_epochs_peak else None
        tracking_df.loc[(metric, class_label, 'Epoch at Convergence'), model_name] = np.mean(valid_epochs_convergence) if valid_epochs_convergence else None

    return tracking_df


def calculate_stats(row, models_names):
    model_values = row[models_names].apply(extract_numeric)
    avg = model_values.mean()
    stddev = model_values.std()
    percent_stddev = (stddev / avg) * 100 if avg != 0 else 0
    return pd.Series({
        'Average': avg,
        'SD': stddev,
        '%SD': percent_stddev,
        'AverageSD Detailed': {'Mean': avg, 'SD': stddev, '%SD': percent_stddev},
        'AverageSD Compact': f"{avg:.2f} ± {stddev:.2f} (± {percent_stddev:.2f}%)"
    })


def extract_numeric(value):
    if isinstance(value, str):
        # Assuming the format is "number - number*" for strings
        numeric_part = value.split('-')[0].strip()  # Split and take the first part
        try:
            return float(numeric_part)
        except ValueError:
            return np.nan  # Return NaN if conversion fails
    elif isinstance(value, (int, float)):
        return float(value)
    return np.nan  # Return NaN for any other unexpected data type


def process_validation_loss(model_dir):
    # Define file path——same for all models
    val_loss_path = os.path.join(model_dir, 'lightning_logs', 'val_loss_step.csv')
    val_loss_data = pd.read_csv(val_loss_path)
    
    # Exclude the first two sanity check entries for epoch 0
    indices_to_drop = val_loss_data[(val_loss_data['Epoch'] == 0) & (val_loss_data['Batch'] < 2)].index[:2]
    val_loss_data = val_loss_data.drop(index=indices_to_drop)
    max_epoch = MAX_EPOCH_NUMBER  # Define the last valid epoch number = 19, 0-start-indexed (epoch 20 is just a repetition of the epoch from the best val loss checkpoint)
    val_loss_data = val_loss_data[val_loss_data['Epoch'] <= max_epoch]
    
    # Calculate the mean validation loss per epoch
    epoch_mean_val_loss = val_loss_data.groupby('Epoch')['Validation Loss'].mean().reset_index()
    
    # Rename columns appropriately
    epoch_mean_val_loss.columns = ['Epoch', 'Average Validation Loss per Epoch']
    
    # Save to a new CSV file and return the DataFrame
    output_path = os.path.join(model_dir, 'lightning_logs', 'average_val_loss_per_epoch_(filtered).csv')
    epoch_mean_val_loss.to_csv(output_path, index=False)
    return epoch_mean_val_loss


def initialise_dataframes(metrics, classes, metrics_actions, models_names):
    """ Initialise and return structured dataframes for storing results and tracking metrics. """
    ## Initialise results dataframes
    results_df_index = pd.MultiIndex.from_product([metrics, classes + ["Macro"]], 
                                                  names=['Metric', 'Class'])
    results_df = pd.DataFrame(columns=models_names + ['Average', 'SD', '%SD', 'AverageSD Detailed', 'AverageSD Compact'],
                              index=results_df_index).astype({'AverageSD Detailed': 'object'})
    
    ## Initialise tracking dataframes
    tracking_df_index = pd.MultiIndex.from_product([metrics, classes + ["All Classes"], metrics_actions], 
                                                   names=['Metric', 'Class', 'Metric Action']).sortlevel()[0]
    tracking_df = pd.DataFrame(columns=models_names + ['Average', 'SD', '%SD', 'AverageSD Detailed', 'AverageSD Compact'],
                              index=tracking_df_index).astype({'AverageSD Detailed': 'object'})

    return results_df, tracking_df


def prepare_and_rearrange_metrics(model_outputs, model_tracked_metrics, tracked_metrics_dir, rearranged_indices, max_epoch_number):
    """
    Prepare target and probability arrays and rearrange tracked metrics for analysis based on new class order.
    
    This function prepares and reorganises the necessary components for:
    - Populating the 'results_df' with metrics calculated using the target and probability arrays.
    - Populating the 'tracking_df' with epochs corresponding to the lowest, peak, and convergence of tracked metrics per class.
    
    Args:
        model_outputs (DataFrame): Contains model output probabilities and targets.
        model_tracked_metrics (DataFrame): Contains metrics tracked over epochs for each class.
        tracked_metrics_dir (str): Directory path where rearranged metrics will be saved.
        rearranged_indices (list): Indices defining the new order of class importance.
        max_epoch_number (int): Maximum epoch number to consider for rearranged metrics.
    
    Returns:
        targets (ndarray): Array shaped (n_samples, n_classes) with binary target values for each class.
        probs (ndarray): Array shaped (n_samples, n_classes) with predicted probabilities for each class.
        model_tracked_metrics_rearranged (DataFrame): DataFrame with metrics rearranged according to new class order.
    
    The 'targets' and 'probs' arrays will be used to calculate the metrics for 'results_df'.
    The 'model_tracked_metrics_rearranged' will be processed to determine epochs for metrics to populate 'tracking_df'.
    """
    ## For results_df: Initialise arrays for storing target labels and probabilities
    n_samples = model_outputs.shape[0]
    n_classes = len(rearranged_indices)
    # targets and probs are of size (n_samples, n_classes) matching expected sklearn.metrics computation parameters
    targets = np.zeros((n_samples, n_classes))
    probs = np.zeros((n_samples, n_classes))

    ## For tracking_df: Initialise DataFrame for rearranged tracked metrics
    n_epochs = model_tracked_metrics.shape[0]
    columns_with_class = [col for col in model_tracked_metrics.columns if 'class' in col]  # Ignoring Epoch and macro metrics
    rearranged_columns = [col.replace('_class', '_rearranged-class') for col in columns_with_class]
    n_columns  = len(columns_with_class) + 1  # Adding a column for the epoch number
    model_tracked_metrics_rearranged = pd.DataFrame(np.zeros((n_epochs, n_columns)), columns=['epoch'] + rearranged_columns)
    model_tracked_metrics_rearranged['epoch'] = np.arange(n_epochs)  # Set 'epoch' column values ranging from 0 to n_epochs-1

    # Split the column names to extract the metric names and class numbers
    tracked_metrics = set()
    tracked_classes = set()
    for col in columns_with_class:
        metric_name, class_number = col.split('_class')
        tracked_metrics.add(metric_name)
        tracked_classes.add(int(class_number))
    # Convert sets to subscriptable lists    
    tracked_metrics = list(tracked_metrics)
    tracked_classes = sorted(list(tracked_classes))

    ## For results_df & tracking_df: 
    # Populate targets and probs arrays, matching rearranged indices following LABELS_BY_RELEVANCE
    for idx, class_idx in enumerate(rearranged_indices):
        probs[:, idx] = model_outputs[f"prob_class_{class_idx+1}"]
        targets[:, idx] = model_outputs[f"target_class_{class_idx+1}"]
        # Rearrange model_tracked_metrics also following the new classes order from LABELS_BY_RELEVANCE
        for tracked_metric in tracked_metrics:
            old_col_name = f"{tracked_metric}_class{class_idx+1}"
            new_col_name = f"{tracked_metric}_rearranged-class{idx+1}"
            model_tracked_metrics_rearranged[new_col_name] = model_tracked_metrics[old_col_name]

    # Save the rearranged tracked metrics in-place, in the model metrics_csv directory
    rearranged_tracked_metrics_path = os.path.join(tracked_metrics_dir, 'Validation - During Training (Rearranged).csv')
    model_tracked_metrics_rearranged = model_tracked_metrics_rearranged[model_tracked_metrics_rearranged['epoch'] <= max_epoch_number]  # 20 epochs in total, from 0 t 19 (ignore epochs no. > 19)
    model_tracked_metrics_rearranged.to_csv(rearranged_tracked_metrics_path, index=False)
    
    return targets, probs, model_tracked_metrics_rearranged


def main_for_zsinfer(args):
    # Retrieve dataset name from configuration based on the provided dataset type
    dataset_name = args.inference + "_on_" + get_dataset_name(args.config)

    # Determine the base output path based on the argument
    if args.save_in_current_dir:
        base_output_path = os.getcwd()  # Use the current working directory for outputs
    else:
        base_output_path = os.path.abspath(os.path.join(args.models_gate_dir, os.pardir))  # Use the parent directory of models_gate_dir
    
    # Setup paths for output directories
    output_dir_name = f"aggregated_seed_results--{args.main_model_name}--{dataset_name}"
    aggregated_seed_results_dir_path = os.path.join(base_output_path, output_dir_name)
    os.makedirs(aggregated_seed_results_dir_path, exist_ok=True)

    # List all directories within the specified path and get the models names
    models_dirs = sorted(
        [os.path.join(args.models_gate_dir, dir) for dir in os.listdir(args.models_gate_dir)
         if os.path.isdir(os.path.join(args.models_gate_dir, dir)) and
            os.path.join(args.models_gate_dir, dir) != aggregated_seed_results_dir_path]
    )
    models_names = [os.path.basename(dir) for dir in models_dirs]

    # rearranged_indices shows the new positions of elements from the original list to match the rearranged order
    rearranged_indices = [LABELS.index(label) for label in LABELS_BY_RELEVANCE]
    # Get focus_labels indices w.r.t. LABELS_BY_RELEVANCE
    focus_labels_indices = [LABELS_BY_RELEVANCE.index(label) for label in args.focus_labels]
    focus_classes = [f"Class {i+1} [{LABELS_BY_RELEVANCE[i]}]" for i in focus_labels_indices]
    other_classes_indices = [i for i in range(len(LABELS_BY_RELEVANCE)) if i not in focus_labels_indices]
    other_classes = [f"Class {i+1} [{LABELS_BY_RELEVANCE[i]}]" for i in other_classes_indices]


    # Initialise dataframes for tracking overall results and detailed metrics.
    results_df, _ = initialise_dataframes(metrics=METRICS, classes=CLASSES, metrics_actions=METRICS_EPOCH_ACTIONS, 
                                                    models_names=models_names)


    # Process each model directory to compute metrics and rearrange data
    for model_dir in models_dirs:
        # Get model name and raw outputs
        model_name = os.path.basename(model_dir)
        outputs_csv_filepath = os.path.join(model_dir, 'outputs_test.csv')
        model_outputs = read_csv_file(outputs_csv_filepath)


        ## Initialise arrays for storing target labels and probabilities
        n_samples = model_outputs.shape[0]
        n_classes = len(rearranged_indices)
        # targets and probs are of size (n_samples, n_classes) matching expected sklearn.metrics computation parameters
        targets = np.zeros((n_samples, n_classes))
        probs = np.zeros((n_samples, n_classes))

        # Populate targets and probs arrays, matching rearranged indices following LABELS_BY_RELEVANCE
        for idx, class_idx in enumerate(rearranged_indices):
            probs[:, idx] = model_outputs[f"prob_class_{class_idx+1}"]
            targets[:, idx] = model_outputs[f"target_class_{class_idx+1}"]


        ## Calculate metrics from raw corresponding targets and probs 
        roc_auc_per_class, _ = calculate_roc_auc(targets=targets, probs=probs)
        pr_auc_per_class, _ = calculate_pr_auc(targets=targets, probs=probs)
        youden_max_per_class, youden_targetfpr_per_class = calculate_youden_index(targets=targets, probs=probs, target_fpr=TARGET_FPR)

        metrics_data = {
            "AUC-ROC": {"value": roc_auc_per_class, "tracked_metric": "auc_roc"},
            "AUC-PR": {"value": pr_auc_per_class, "tracked_metric": "auc_pr"},
            "Maximum Youden\'s J Statistic": {"value": youden_max_per_class, "tracked_metric": "j_index_max"},
            f"Youden\'s J Statistic at {int(TARGET_FPR*100)}% FPR": {"value": youden_targetfpr_per_class, "tracked_metric": "j_index_fpr"}
        }

        ## Iterate over each type of metric to populate results and tracking dataframes.   
        for metric, data in metrics_data.items():
            # Populate results and tracking dataframes for each class.
            for idx, value in enumerate(data["value"]):
                results_df.loc[(metric, CLASSES[idx]), model_name] = value  # Record metric values for each class.
            # Aggregate and record macro metric values across all classes.
            results_df.loc[(metric, "Macro"), model_name] = np.mean(data["value"])
        

    # Add average and standard deviation across models for each metric-class for results_df
    results_df[['Average', 'SD', '%SD', 'AverageSD Detailed', 'AverageSD Compact']] = results_df.apply(lambda row: calculate_stats(row, models_names), axis=1)
    # Modify results dataframe to focus on most relevant classes
    focused_results_df = modify_results_for_focus_labels(results_df=results_df, focus_classes=focus_classes, other_classes=other_classes, 
                                                         metrics=METRICS, models_names=models_names, average_focus_class_name=AVERAGE_FOCUS_CLASS_NAME,
                                                         others_class_name=OTHERS_CLASS_NAME)

    ## Final structuring: resetting the index of the DataFrame to convert the multi-level index into columns
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'level_0': 'Metric', 'level_1': 'Class'}, inplace=True)

    ## Saving the detailed and focused results_df to CSV files
    results_df_outpath = os.path.join(aggregated_seed_results_dir_path, f'aggregated_results_metrics_detailed--{args.main_model_name}--{dataset_name}.csv')
    results_df.to_csv(results_df_outpath, index=False)
    results_df_outpath_focused = os.path.join(aggregated_seed_results_dir_path, f'aggregated_results_metrics_focused--{args.main_model_name}--{dataset_name}.csv')
    focused_results_df.to_csv(results_df_outpath_focused, index=False)


def main(args):
    # Retrieve dataset name from configuration based on the provided dataset type
    dataset_name = get_dataset_name(args.config)
    if args.inference:
        dataset_name = args.inference + "_on_" + dataset_name

    # Determine the base output path based on the argument
    if args.save_in_current_dir:
        base_output_path = os.getcwd()  # Use the current working directory for outputs
    else:
        base_output_path = os.path.abspath(os.path.join(args.models_gate_dir, os.pardir))  # Use the parent directory of models_gate_dir
    
    # Setup paths for output directories
    output_dir_name = f"aggregated_seed_results--{args.main_model_name}--{dataset_name}"
    aggregated_seed_results_dir_path = os.path.join(base_output_path, output_dir_name)
    os.makedirs(aggregated_seed_results_dir_path, exist_ok=True)

    # List all directories within the specified path and get the models names
    models_dirs = sorted(
        [os.path.join(args.models_gate_dir, dir) for dir in os.listdir(args.models_gate_dir)
         if os.path.isdir(os.path.join(args.models_gate_dir, dir)) and
            os.path.join(args.models_gate_dir, dir) != aggregated_seed_results_dir_path]
    )
    models_names = [os.path.basename(dir) for dir in models_dirs]

    # rearranged_indices shows the new positions of elements from the original list to match the rearranged order
    rearranged_indices = [LABELS.index(label) for label in LABELS_BY_RELEVANCE]
    # Get focus_labels indices w.r.t. LABELS_BY_RELEVANCE
    focus_labels_indices = [LABELS_BY_RELEVANCE.index(label) for label in args.focus_labels]
    focus_classes = [f"Class {i+1} [{LABELS_BY_RELEVANCE[i]}]" for i in focus_labels_indices]
    other_classes_indices = [i for i in range(len(LABELS_BY_RELEVANCE)) if i not in focus_labels_indices]
    other_classes = [f"Class {i+1} [{LABELS_BY_RELEVANCE[i]}]" for i in other_classes_indices]


    # Initialise dataframes for tracking overall results and detailed metrics.
    results_df, tracking_df = initialise_dataframes(metrics=METRICS, classes=CLASSES, metrics_actions=METRICS_EPOCH_ACTIONS, 
                                                    models_names=models_names)


    # Process each model directory to compute metrics and rearrange data
    for model_dir in models_dirs:
        # Get model name and raw outputs
        model_name = os.path.basename(model_dir)
        outputs_csv_filepath = os.path.join(model_dir, 'outputs_test.csv')
        model_outputs = read_csv_file(outputs_csv_filepath)

        # Get model tracked metrics
        tracked_metrics_dir = os.path.join(model_dir, 'metrics_csv')
        tracked_metrics_path = os.path.join(tracked_metrics_dir, 'Validation - During Training.csv')
        model_tracked_metrics = read_csv_file(tracked_metrics_path)

        # Get model checkpoint epoch
        checkpoint_dir = os.path.join(model_dir, 'lightning_checkpoints')
        epoch_at_min_val_loss = extract_epoch_from_checkpoint(checkpoint_dir)

        # Process validation losses and populate tracking_df with the epoch values
        average_val_loss_df = process_validation_loss(model_dir)
        val_losses = average_val_loss_df['Average Validation Loss per Epoch'].tolist()
        tracking_df = populate_tracking_df(
            metric_values=val_losses, 
            metric='Validation Loss', 
            class_label='All Classes',
            model_name=model_name, 
            tracking_df=tracking_df, 
            epoch_from_checkpoint=epoch_at_min_val_loss
        )

        ## Get the 'targets' and 'probs' to be used to calculate the metrics for 'results_df'.
        # As well as the 'model_tracked_metrics_rearranged' to be processed to determine epochs for metrics to populate 'tracking_df'.
        targets, probs, model_tracked_metrics_rearranged = prepare_and_rearrange_metrics(
            model_outputs=model_outputs, 
            model_tracked_metrics=model_tracked_metrics, 
            tracked_metrics_dir=tracked_metrics_dir, 
            rearranged_indices=rearranged_indices, 
            max_epoch_number=MAX_EPOCH_NUMBER
            )

        ## Calculate metrics from raw corresponding targets and probs 
        roc_auc_per_class, _ = calculate_roc_auc(targets=targets, probs=probs)
        pr_auc_per_class, _ = calculate_pr_auc(targets=targets, probs=probs)
        youden_max_per_class, youden_targetfpr_per_class = calculate_youden_index(targets=targets, probs=probs, target_fpr=TARGET_FPR)

        metrics_data = {
            "AUC-ROC": {"value": roc_auc_per_class, "tracked_metric": "auc_roc"},
            "AUC-PR": {"value": pr_auc_per_class, "tracked_metric": "auc_pr"},
            "Maximum Youden\'s J Statistic": {"value": youden_max_per_class, "tracked_metric": "j_index_max"},
            f"Youden\'s J Statistic at {int(TARGET_FPR*100)}% FPR": {"value": youden_targetfpr_per_class, "tracked_metric": "j_index_fpr"}
        }
        
        # Define a global dictionary to collect metrics epoch data for averaging
        all_classes_epoch_results = {metric: {action: [] for action in METRICS_EPOCH_ACTIONS} for metric in METRICS}

        ## Iterate over each type of metric to populate results and tracking dataframes.   
        for metric, data in metrics_data.items():
            # Populate results and tracking dataframes for each class.
            for idx, value in enumerate(data["value"]):
                results_df.loc[(metric, CLASSES[idx]), model_name] = value  # Record metric values for each class.
                tracked_values = model_tracked_metrics_rearranged[f'{data["tracked_metric"]}_rearranged-class{idx+1}'].tolist()
                tracking_df = populate_tracking_df(metric_values=tracked_values, metric=metric, class_label=CLASSES[idx], 
                                                   model_name=model_name, tracking_df=tracking_df, 
                                                   all_classes_epoch_results=all_classes_epoch_results)
            # Aggregate and record macro metric values across all classes.
            results_df.loc[(metric, "Macro"), model_name] = np.mean(data["value"])
            tracking_df = populate_tracking_df(metric_values=tracked_values, metric=metric, class_label="All Classes", 
                                               model_name=model_name, tracking_df=tracking_df, 
                                               all_classes_epoch_results=all_classes_epoch_results)
        

    # Add average and standard deviation across models for each metric-class for results_df
    results_df[['Average', 'SD', '%SD', 'AverageSD Detailed', 'AverageSD Compact']] = results_df.apply(lambda row: calculate_stats(row, models_names), axis=1)
    # Modify results dataframe to focus on most relevant classes
    focused_results_df = modify_results_for_focus_labels(results_df=results_df, focus_classes=focus_classes, other_classes=other_classes, 
                                                         metrics=METRICS, models_names=models_names, average_focus_class_name=AVERAGE_FOCUS_CLASS_NAME,
                                                         others_class_name=OTHERS_CLASS_NAME)
    
    # Add average and standard deviation across models for each metric-class-metric_action for tracking_df
    tracking_df[['Average', 'SD', '%SD', 'AverageSD Detailed', 'AverageSD Compact']] = tracking_df.apply(lambda row: calculate_stats(row, models_names), axis=1)
    # Modify tracking dataframe to focus on most relevant classes
    custom_tracking_order_focused = get_custom_tracking_order_focused(focus_classes=focus_classes, average_focus_class_name=AVERAGE_FOCUS_CLASS_NAME,
                                                                      others_class_name=OTHERS_CLASS_NAME)
    focused_tracking_df = modify_tracking_metrics_for_focus_labels(tracking_df=tracking_df, focus_classes=focus_classes, other_classes=other_classes, 
                                                                   metrics=METRICS, models_names=models_names, metrics_actions=METRICS_EPOCH_ACTIONS,
                                                                   custom_tracking_order_focused=custom_tracking_order_focused, 
                                                                   average_focus_class_name=AVERAGE_FOCUS_CLASS_NAME, others_class_name=OTHERS_CLASS_NAME)


    ## Final structuring: resetting the index of the DataFrame to convert the multi-level index into columns
    # results_df:
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'level_0': 'Metric', 'level_1': 'Class'}, inplace=True)
    # tracking_df:
    tracking_df = tracking_df.reindex(pd.MultiIndex.from_tuples(CUSTOM_TRACKING_ORDER_DETAILED, names=['Metric', 'Class', 'Metric Action']))  # reordering of the rows after initial sort for optimised indexing
    tracking_df.reset_index(inplace=True)
    tracking_df.rename(columns={'level_0': 'Metric', 'level_1': 'Class', 'level_2': 'Metric Action'}, inplace=True)


    ## Saving the detailed and focused results_df to CSV files
    results_df_outpath = os.path.join(aggregated_seed_results_dir_path, f'aggregated_results_metrics_detailed--{args.main_model_name}--{dataset_name}.csv')
    results_df.to_csv(results_df_outpath, index=False)
    results_df_outpath_focused = os.path.join(aggregated_seed_results_dir_path, f'aggregated_results_metrics_focused--{args.main_model_name}--{dataset_name}.csv')
    focused_results_df.to_csv(results_df_outpath_focused, index=False)
    ## Saving the detailed and focused tracking_df to CSV files
    tracking_df_outpath = os.path.join(aggregated_seed_results_dir_path, f'aggregated_tracking_metrics_detailed--{args.main_model_name}--{dataset_name}.csv')
    tracking_df.to_csv(tracking_df_outpath, index=False)
    tracking_df_outpath_focused = os.path.join(aggregated_seed_results_dir_path, f'aggregated_tracking_metrics_focused--{args.main_model_name}--{dataset_name}.csv')
    focused_tracking_df.to_csv(tracking_df_outpath_focused, index=False)

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# ========= PERFORMANCE METRICS COMPUTATION - END ========
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<



def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate and compute performance metrics across multiple model runs.")
    parser.add_argument('--models_gate_dir', required=True, help='Path to multiruns directory of a single model we want to aggregate the results from.')
    parser.add_argument('--main_model_name', required=True, help='Main model name identifier for output labeling.')
    parser.add_argument('--config', default='chexpert', choices=['chexpert', 'mimic'], help='Config dataset module to use.')
    parser.add_argument('--focus_labels', nargs='+', default=["Pleural Effusion", "No Finding", "Cardiomegaly", "Pneumothorax",
                                                              "Atelectasis", "Consolidation", "Edema"], help='List of most important labels.')
    parser.add_argument('--save_in_current_dir', action='store_true', help='Flag to save outputs in the current directory instead of the base model directory.')
    parser.add_argument('--inference', default=False, choices=['ZSInfer', 'LPInfer', 'FFTInfer'], help='Specify the inference model to prefix dataset name for specialised output.')
    return parser.parse_args()



if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    if args.inference == 'ZSInfer':
        main_for_zsinfer(args)
    else: 
        main(args)