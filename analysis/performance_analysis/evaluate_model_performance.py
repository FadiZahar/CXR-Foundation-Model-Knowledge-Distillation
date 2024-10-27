import os
import argparse
import numpy as np
import pandas as pd

# Import global shared variables
from config.config_shared import MODEL_STYLES

# Import other utils functions
from analysis.bias_analysis.evaluate_models_disease_prediction import read_csv_file

np.random.seed(42)



def get_model_info(seed_results_dirpath):
    dirname = os.path.basename(seed_results_dirpath)
    parts = dirname.split("--")
    model_shortname = parts[1]
    dataset_name = parts[-1]
    return model_shortname, dataset_name


def load_aggregated_results(directory, model_shortname, dataset_name):
    """Load aggregated results metrics from specified directory."""
    detailed_path = os.path.join(directory, f'aggregated_results_metrics_detailed--{model_shortname}--{dataset_name}.csv')
    focused_path = os.path.join(directory, f'aggregated_results_metrics_focused--{model_shortname}--{dataset_name}.csv')
    if os.path.exists(detailed_path):
        detailed_df = read_csv_file(detailed_path)
    else:
        detailed_df = None
    if os.path.exists(focused_path):
        focused_df = read_csv_file(focused_path)
    else:
        focused_df = None
    return detailed_df, focused_df


def parse_args():
    parser = argparse.ArgumentParser(description="Plot performance metrics from aggregated results.")
    parser.add_argument('--models_seed_results_dirpath', nargs='+', required=True,
                        help="Paths to directories with aggregated seed results.")
    return parser.parse_args()



if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # List to store models info and data
    models = []
    for seed_results_dirpath in args.models_seed_results_dirpath:
        model_shortname, dataset_name = get_model_info(seed_results_dirpath)
        detailed_df, focused_df = load_aggregated_results(directory=seed_results_dirpath, model_shortname=model_shortname, dataset_name=dataset_name)
        models.append({
            "seed_results_dirpath": seed_results_dirpath,
            "shortname": model_shortname,
            "dataset_name": dataset_name,
            "color": MODEL_STYLES[model_shortname]['color'],
            "marker": MODEL_STYLES[model_shortname]['marker'],
            "linestyle": MODEL_STYLES[model_shortname]['linestyle'],
            "detailed_df": detailed_df,
            "focused_df": focused_df
        })