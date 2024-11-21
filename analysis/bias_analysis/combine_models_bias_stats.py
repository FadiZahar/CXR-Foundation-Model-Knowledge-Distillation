import os
import argparse
import pandas as pd

# Import the configuration loader
from config.loader_config import load_config, get_dataset_name




def list_subdirectories(directory):
    """List all subdirectories in a given directory."""
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]


def get_csv_file_patterns(directory):
    """Get unique CSV file patterns by removing the model-specific prefix."""
    patterns = {}
    for file in os.listdir(directory):
        if file.endswith('.csv'):
            # Split the filename to remove the model-specific (model name) part and use the rest as the pattern
            parts = file.split('__', 1)
            if len(parts) > 1:
                pattern = parts[1]
                if pattern not in patterns:
                    patterns[pattern] = []
                patterns[pattern].append(os.path.join(directory, file))
    return patterns


def concatenate_csv_files(file_paths):
    """Concatenate all CSV files in the file_paths list into a single DataFrame."""
    df_list = [pd.read_csv(f) for f in file_paths]
    return pd.concat(df_list, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Model bias scores extraction.")
    parser.add_argument('--config', default='chexpert', choices=['chexpert', 'mimic'], help='Config dataset module to use')
    parser.add_argument('--models_bias_stats_dirpath', nargs='+', default=None, 
                        help="Provide one or more models' bias_stats directory paths containing the bias statistics CSV files.")
    parser.add_argument('--bias2', action='store_true', 
                        help='Enable additional bias analysis mode.')
    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()
    # Load the configuration dynamically based on the command line argument
    config = load_config(args.config)
    # Accessing the configuration to import dataset-specific variables
    dataset_name = get_dataset_name(args.config)

    # Path to base output directory
    base_output_path = os.getcwd()  # Use the current working directory for outputs

    # Determine the directory name based on whether --bias2 is specified
    # i.e., change to f'bias_inspection--{dataset_name}' to correspond with evaluate_model_bias.py rather than evaluate_model_bias2.py
    directory_name_suffix = f'bias_inspection2--{dataset_name}' if args.bias2 else f'bias_inspection--{dataset_name}'

    # 'Global' directories to save bias stats outputs
    global_bias_inspection_dir_path = os.path.join(base_output_path, directory_name_suffix)
    global_bias_stats_dir_path = os.path.join(global_bias_inspection_dir_path, 'models_bias_stats/')
    os.makedirs(global_bias_stats_dir_path, exist_ok=True)


    ## Process the first directory to establish the structure
    first_dir_path = args.models_bias_stats_dirpath[0]
    subdirectories = list_subdirectories(first_dir_path)
    csv_patterns = {}

    for subdir in subdirectories:
        subdir_path = os.path.join(first_dir_path, subdir)
        patterns = get_csv_file_patterns(subdir_path)
        for pattern, matching_file_paths in patterns.items():
            csv_patterns[(subdir, pattern)] = matching_file_paths

    
    ## Process remaining directories using established patterns
    for dir_path in args.models_bias_stats_dirpath[1:]:
        for (subdir, pattern) in csv_patterns.keys():
            subdir_path = os.path.join(dir_path, subdir)
            if os.path.exists(subdir_path):
                new_matching_file_paths = get_csv_file_patterns(subdir_path).get(pattern, [])
                csv_patterns[(subdir, pattern)].extend(new_matching_file_paths)


    ## Concatenate and save CSVs per pattern
    for (subdir, pattern), files in csv_patterns.items():
        if files:
            final_df = concatenate_csv_files(files)
            output_subdir = os.path.join(global_bias_stats_dir_path, subdir)
            os.makedirs(output_subdir, exist_ok=True)
            output_file = os.path.join(output_subdir, f"All-Models__{pattern}")
            final_df.to_csv(output_file, index=False)
            print(f"Concatenated DataFrame saved to {output_file}")



