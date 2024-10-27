import os
import argparse
import numpy as np
import pandas as pd

# Import utils functions
from analysis.bias_analysis.evaluate_models_disease_prediction import read_csv_file

# Global Variables
STATS_BOOLEANS = ['FALSE', 'TRUE', 'TRUE+']
MAX_STATS_BOOLEAN_LENGTH = max(len(stats_bool) for stats_bool in STATS_BOOLEANS)
WEIGHTS_STATS_BOOLEANS = {
    STATS_BOOLEANS[0]: 0, 
    STATS_BOOLEANS[1]: 100, 
    STATS_BOOLEANS[2]: 150
    }
COLUMNS_OF_INTEREST = ['White vs Asian', 'White vs Black', 'Asian vs Black', 'Male vs Female']
WEIGHTS_COLUMNS = { 
    COLUMNS_OF_INTEREST[0]: 1.0, 
    COLUMNS_OF_INTEREST[1]: 1.0, 
    COLUMNS_OF_INTEREST[2]: 1.0,
    COLUMNS_OF_INTEREST[3]: 1.0
    }  # We assume equal weighting for each column (whether comparison was between race or sex subgroups)
ATTRIBUTES = {'Race': [COLUMNS_OF_INTEREST[0], COLUMNS_OF_INTEREST[1], COLUMNS_OF_INTEREST[2]], 'Sex': [COLUMNS_OF_INTEREST[3]]}
WEIGHTS_ATTRIBUTES = { 
    'Race': 1.0, 
    'Sex': 1.0,
    }  # We assume equal weighting for each attribute



def parse_args():
    parser = argparse.ArgumentParser(description="Model bias scores extraction.")
    parser.add_argument('--model_bias_csv_path', required=True, help="Path to the model's bias csv file")
    return parser.parse_args()


def extract_bias_counts(string):
    # Extract counts from formatted string
    a = [x.split(":") for x in string.split('\n')]
    return {b.strip(): int(c.strip().split(" ")[0]) for b, c in a}


def get_combined_bias_score(bias_scores_dict):
    combined_score = sum(WEIGHTS_STATS_BOOLEANS[stats_bool] * bias_scores_dict[stats_bool] for stats_bool in STATS_BOOLEANS)
    return combined_score


def should_consider_row(pe_counts):
    # Only consider rows where TRUE + TRUE+ > FALSE
    total_true = pe_counts['TRUE'] + pe_counts['TRUE+']
    return total_true > pe_counts['FALSE']


def get_output_filename(model_bias_csv_path, filtered_data=False, compact_df=False):
    # Generate output filename based on input path
    n = os.path.basename(model_bias_csv_path)
    suffix1 = "--filtered" if filtered_data else ""
    suffix2 = "--compact" if compact_df else ""
    n1 = n.split("--")[0]
    n2 = n.split("--")[-1].split("__")[-1]
    output_filename = f"{n1}--compiled_bias_scores{suffix1}{suffix2}__{n2}"
    return output_filename


def append_scores(source_dict, row_entries, normalised=True, total_explained_variance=1):
    """ Append bias scores, formatted bias scores, and combined bias score to row_entries."""

    # Filter source_dict to include only keys in STATS_BOOLEANS, excluding 'Combined Bias Score'
    filtered_scores = {stats_bool: source_dict[stats_bool] for stats_bool in STATS_BOOLEANS}

    if normalised:
        row_entries.extend([
            filtered_scores,  # Bias scores
            "\n".join(f"{stats_bool:{MAX_STATS_BOOLEAN_LENGTH}} : {score*100:.2f}%" for stats_bool, score in filtered_scores.items()),  # Formatted bias scores
            source_dict['Combined Bias Score']  # Combined bias score
        ])
    else:
        source_dict_unnormalised = {stats_bool: score * total_explained_variance for stats_bool, score in filtered_scores.items()}
        row_entries.extend([
            source_dict_unnormalised,  # Bias scores -- Unnormalised
            "\n".join(f"{stats_bool:{MAX_STATS_BOOLEAN_LENGTH}} : {score*100:.2f}%" for stats_bool, score in source_dict_unnormalised.items()),  # Formatted bias scores -- Unnormalised
            source_dict['Combined Bias Score'] * total_explained_variance  # Combined bias score -- Unnormalised
        ])


def get_columns_names(string, scores_naming_patterns, normalised=True):
    names = []
    suffix = "" if normalised else " -- Unnormalised"
    for score_naming_pattern in scores_naming_patterns:
        names.append(f"{string} ({score_naming_pattern}{suffix})")
    return names


def extract_bias_scores(model_bias_csv_path, data, filtered_data=False):
    # Setup output file paths in the specified output directory
    output_dir = os.path.dirname(model_bias_csv_path)
    output_filename = get_output_filename(model_bias_csv_path, filtered_data=filtered_data)
    output_file_path = os.path.join(output_dir, output_filename)
    compact_output_filename = get_output_filename(model_bias_csv_path, filtered_data=filtered_data, compact_df=True)
    compact_output_file_path = os.path.join(output_dir, compact_output_filename)

    # Initialise lists to store detailed and compact results
    results = []

    # Iterate through each unique statistical test applied to the data
    for test in data['Statistical Test Applied'].unique():
        test_data = data[data['Statistical Test Applied'] == test]
        if filtered_data:
            # # Filter to only consider PCA Modes which capture differences in the features related to presence of disease (indicated by the significant differences between "no finding" and "pleural effusion")
            # test_data = test_data[test_data['Pleural Effusion vs No Finding'].apply(lambda x: should_consider_row(extract_bias_counts(x)))]
            
            # Filter to only consider rows from PCA Mode 1
            test_data = test_data[test_data['Mode'] == 'PCA Mode 1']

        # Initialise dictionaries to store bias scores per column and attribute
        bias_scores_test_per_column = {column : {stats_bool: 0 for stats_bool in STATS_BOOLEANS} for column in COLUMNS_OF_INTEREST}
        bias_scores_test_per_attribute = {attribute : {stats_bool: 0 for stats_bool in STATS_BOOLEANS} for attribute in ATTRIBUTES}

        total_weight_columns = sum(WEIGHTS_COLUMNS.values())
        total_weight_attributes = sum(WEIGHTS_ATTRIBUTES.values())

        explained_variances = test_data['Explained Variance']
        total_explained_variance = explained_variances.sum()
        normalised_explained_variances = explained_variances/total_explained_variance

        # Aggregate and compute bias scores for each column
        for column in COLUMNS_OF_INTEREST:
            entries = [extract_bias_counts(entry) for entry in test_data[column]]
            for entry, norm_var in zip (entries, normalised_explained_variances):
                total_count = sum(entry.values())
                if total_count > 0:
                    for stats_bool in STATS_BOOLEANS:
                        stats_bool_percentage = (entry[stats_bool] / total_count)
                        bias_scores_test_per_column[column][stats_bool] += stats_bool_percentage * norm_var

        # Calculate weighted averages and combined bias scores for columns
        bias_scores_test_per_column['Average'] = {
            stats_bool: sum(WEIGHTS_COLUMNS[column] * bias_scores_test_per_column[column][stats_bool] for column in COLUMNS_OF_INTEREST)/total_weight_columns 
            for stats_bool in STATS_BOOLEANS
            }
        for column in COLUMNS_OF_INTEREST:
           bias_scores_test_per_column[column]['Combined Bias Score'] = get_combined_bias_score(bias_scores_test_per_column[column])
        bias_scores_test_per_column['Average']['Combined Bias Score'] = get_combined_bias_score(bias_scores_test_per_column['Average'])


        # Calculate and update bias scores for attributes
        for attribute in ATTRIBUTES:
            for stats_bool in STATS_BOOLEANS:
                bias_scores_test_per_attribute[attribute][stats_bool] = np.mean([bias_scores_test_per_column[column][stats_bool] for column in ATTRIBUTES[attribute]])
        # Calculate weighted averages and combined bias scores for attributes
        bias_scores_test_per_attribute['Average'] = {
            stats_bool: sum(WEIGHTS_ATTRIBUTES[attribute] * bias_scores_test_per_attribute[attribute][stats_bool] for attribute in ATTRIBUTES)/total_weight_attributes 
            for stats_bool in STATS_BOOLEANS
            }
        for attribute in ATTRIBUTES:
           bias_scores_test_per_attribute[attribute]['Combined Bias Score'] = get_combined_bias_score(bias_scores_test_per_attribute[attribute])
        bias_scores_test_per_attribute['Average']['Combined Bias Score'] = get_combined_bias_score(bias_scores_test_per_attribute['Average'])
        

        # Assemble results for each statistical test
        result_test = [
            data['Model Fullname'].iloc[0],  # Ensure all entries pertain to the same model
            data['Model Shortname'].iloc[0],  
            test
            ]

        # Append normalised and unnormalised scores for columns and attributes to results
        for is_normalised in [True, False]:
            for column in COLUMNS_OF_INTEREST:
                append_scores(bias_scores_test_per_column[column], result_test, normalised=is_normalised, total_explained_variance=total_explained_variance)
            append_scores(bias_scores_test_per_column['Average'], result_test, normalised=is_normalised, total_explained_variance=total_explained_variance)
            for attribute in ATTRIBUTES:
                append_scores(bias_scores_test_per_attribute[attribute], result_test, normalised=is_normalised, total_explained_variance=total_explained_variance)
            append_scores(bias_scores_test_per_attribute['Average'], result_test, normalised=is_normalised, total_explained_variance=total_explained_variance)

        results.append(result_test)


    # Create DataFrame
    scores_naming_patterns = ['Bias Scores', 'Formatted Bias Scores', 'Combined Bias Score']
    df_columns_names = ['Model Fullname', 'Model Shortname', 'Statistical Test Applied']

    # Dynamically generate column names for both normalised and unnormalised data for all metrics
    for is_normalised in [True, False]:
        for column in COLUMNS_OF_INTEREST:
            df_columns_names.extend(get_columns_names(string=column, scores_naming_patterns=scores_naming_patterns, normalised=is_normalised))
        df_columns_names.extend(get_columns_names(string='Columns Average', scores_naming_patterns=scores_naming_patterns, normalised=is_normalised))
        for attribute in ATTRIBUTES:
            df_columns_names.extend(get_columns_names(string=attribute, scores_naming_patterns=scores_naming_patterns, normalised=is_normalised))
        df_columns_names.extend(get_columns_names(string='Attributes Average', scores_naming_patterns=scores_naming_patterns, normalised=is_normalised))

    # Initialise and populate the DataFrame with results, using the dynamically generated column names
    results_df = pd.DataFrame(results, columns=df_columns_names)

    # Generate a list of column names that include '(Bias Scores)' but not '(Formatted Bias Scores)'
    columns_to_drop = [col for col in results_df.columns if any(term in col for term in COLUMNS_OF_INTEREST) or '(Bias Scores' in col]
    # Create a compact DataFrame by dropping the specified columns
    results_df_compact = results_df.drop(columns=columns_to_drop)


    # Save both DataFrames to CSV files
    results_df.to_csv(output_file_path, index=False)
    results_df_compact.to_csv(compact_output_file_path, index=False) 

    print(f"Results saved to {output_file_path}")
    print(f"Compact results saved to {compact_output_file_path}")



if __name__ == "__main__":

    args = parse_args()
    # Load data
    data = read_csv_file(args.model_bias_csv_path)

    # Extract bias scores across all PCA modes to assess overall model behavior without filtering.
    extract_bias_scores(model_bias_csv_path=args.model_bias_csv_path, data=data, filtered_data=False)

    # Extract bias scores for the first PCA mode only showing significant disease detection differences (Pleural Effusion vs No Finding).
    extract_bias_scores(model_bias_csv_path=args.model_bias_csv_path, data=data, filtered_data=True)




