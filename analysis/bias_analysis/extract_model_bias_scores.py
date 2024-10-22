import os
import argparse
import numpy as np
import pandas as pd

# Import utils functions
from analysis.bias_analysis.evaluate_models_disease_prediction import read_csv_file

# Global Variables
STATS_BOOLEANS = ['FALSE', 'TRUE', 'TRUE+']
WEIGHTS = {
    STATS_BOOLEANS[0]: 0, 
    STATS_BOOLEANS[1]: 100, 
    STATS_BOOLEANS[2]: 150
    }



def parse_args():
    parser = argparse.ArgumentParser(description="Model bias scores extraction.")
    parser.add_argument('--model_bias_csv_path', required=True, help="Path to the model's bias csv file")
    return parser.parse_args()


def extract_bias_counts(string):
    # Extract counts from formatted string
    a = [x.split(":") for x in string.split('\n')]
    return {b.strip(): int(c.strip().split(" ")[0]) for b, c in a}


def should_consider_row(pe_counts):
    # Only consider rows where TRUE + TRUE+ > FALSE
    total_true = pe_counts['TRUE'] + pe_counts['TRUE+']
    return total_true > pe_counts['FALSE']


def get_output_filename(model_bias_csv_path, filtered_data=False):
    # Generate output filename based on input path
    n = os.path.basename(model_bias_csv_path)
    n1 = n.split("--")[0]
    n2 = n.split("--")[-1].split("__")[-1]
    output_filename = f"{n1}--compiled_bias_scores--filtered__{n2}" if filtered_data else f"{n1}--compiled_bias_scores__{n2}"
    return output_filename


def extract_bias_scores(model_bias_csv_path, data, filtered_data=False):
    # Prepare output directory
    output_dir = os.path.dirname(model_bias_csv_path)
    output_filename = get_output_filename(model_bias_csv_path, filtered_data=filtered_data)
    output_file_path = os.path.join(output_dir, output_filename)

    # Initialise results list
    results = []

    for test in data['Statistical Test Applied'].unique():
        test_data = data[data['Statistical Test Applied'] == test]
        if filtered_data:
            test_data = test_data[test_data['Pleural Effusion vs No Finding'].apply(lambda x: should_consider_row(extract_bias_counts(x)))]

        bias_score_test = {stats_bool: [] for stats_bool in STATS_BOOLEANS}

        explained_variances = test_data['Explained Variance']
        total_explained_variance = explained_variances.sum()
        normalised_explained_variances = explained_variances/total_explained_variance

        columns_of_interest = ['White vs Asian', 'White vs Black', 'Asian vs Black', 'Male vs Female']

        for column in columns_of_interest:
            entries = [extract_bias_counts(entry) for entry in test_data[column]]
            bias_test_score_col = {stats_bool: 0 for stats_bool in STATS_BOOLEANS}

            for entry, norm_var in zip (entries, normalised_explained_variances):
                total_count = sum(entry.values())
                if total_count > 0:
                    for stats_bool in STATS_BOOLEANS:
                        bias_test_score_col[stats_bool] += (entry[stats_bool] / total_count) * norm_var

            for stats_bool in STATS_BOOLEANS:
                bias_score_test[stats_bool].append(bias_test_score_col[stats_bool])

        for stats_bool in STATS_BOOLEANS:
            bias_score_test[stats_bool] = np.mean(bias_score_test[stats_bool])  # We assume equal weighting for each column (whether comparison was between race or sex subgroups)

        # Prepare row for DataFrame
        max_stats_bool_length = max(len(stats_bool) for stats_bool in STATS_BOOLEANS)
        scores_formatted = "\n".join(f"{stats_bool:{max_stats_bool_length}} : {score*100:.2f}%" for stats_bool, score in bias_score_test.items())
        combined_score = sum(WEIGHTS[stat] * bias_score_test[stat] for stat in STATS_BOOLEANS)
        scores_unnormalised = {stat: score * total_explained_variance for stat, score in bias_score_test.items()}
        scores_unnormalised_formatted = "\n".join(f"{stats_bool:{max_stats_bool_length}} : {score*100:.2f}%" for stats_bool, score in scores_unnormalised.items())
        combined_score_unnormalised = combined_score * total_explained_variance

        results.append([
            data['Model Fullname'].iloc[0],  # all rows have the same model name
            data['Model Shortname'].iloc[0],  # all rows have the same short name
            test,
            bias_score_test,
            scores_formatted,
            combined_score,
            scores_unnormalised,
            scores_unnormalised_formatted,
            combined_score_unnormalised
        ])

    # Create DataFrame
    results_df = pd.DataFrame(results, columns=[
        'Model Fullname', 'Model Shortname', 'Statistical Test Applied', 'Bias Scores',
        'Formatted Bias Scores', 'Combined Bias Score', 'Bias Scores -- Unnormalised',
        'Formatted Bias Scores -- Unnormalised', 'Combined Bias Score -- Unnormalised'
    ])

    # Save to CSV
    results_df.to_csv(output_file_path, index=False)
    print(f"Results saved to {output_file_path}")




if __name__ == "__main__":

    args = parse_args()
    # Load data
    data = read_csv_file(args.model_bias_csv_path)

    # Extract bias scores across all PCA modes to assess overall model behavior without filtering.
    extract_bias_scores(model_bias_csv_path=args.model_bias_csv_path, data=data, filtered_data=False)

    # Extract bias scores for PCA modes showing significant disease detection differences (Pleural Effusion vs No Finding).
    # This focuses the analysis on modes where disease presence significantly alters model outputs, indicating key discriminative features.
    extract_bias_scores(model_bias_csv_path=args.model_bias_csv_path, data=data, filtered_data=True)




