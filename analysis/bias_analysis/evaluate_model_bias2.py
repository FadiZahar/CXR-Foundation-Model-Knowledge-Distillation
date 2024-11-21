import os
import argparse
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from scipy.stats import ks_2samp, mannwhitneyu, anderson_ksamp, median_test, mood, kruskal, cramervonmises_2samp
from scipy.stats import combine_pvalues
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
from matplotlib import font_manager
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg


# Check if Latin Modern Roman (~LaTeX) is available, and set it; otherwise, use the default font
if 'Latin Modern Roman' in [f.name for f in font_manager.fontManager.ttflist]:
    plt.style.use('default')
    plt.rcParams['font.family'] = 'Latin Modern Roman'


# Import global shared variables
from config.config_shared import OUT_DPI, RACES, SEXES
# Import the configuration loader
from config.loader_config import load_config, get_dataset_name
# Import utils functions
from analysis.bias_analysis.evaluate_models_disease_prediction import read_csv_file

# Define other global variables
OUT_FORMAT = 'png'
RASTERIZED_SCATTER = True
N_SAMPLES = 1000
N_SIMULATION_ITERATIONS = 5000
N_PERMUTATIONS = 10000
STRATA_COLUMNS = ["sex", "disease", "age_bin"]

ALPHA = 0.6
MARKER = 'o'
MARKERSIZE = 40
FONT_SCALE = 1.25
COLOR_PALETTE1 = ['deepskyblue', 'darkorange', 'forestgreen', 'darkorchid', 'red']
COLOR_PALETTE2 = 'plasma_r'
KIND = 'scatter'
STD_SCALE = 3

AGE_BINS = {
    '0-20': (0, 20),
    '21-30': (21, 30),
    '31-40': (31, 40),
    '41-50': (41, 50),
    '51-60': (51, 60),
    '61-70': (61, 70),
    '71-80': (71, 80),
    '81+': (81, float('inf'))
}

MAIN_SAMPLE_SUBDIR_NAME='main_sample/'
ALL_UNIQUE_DATA_SUBDIR_NAME='all_unique_data_points/'
SIMULATION_SUBDIR_NAME='simulation/'

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)




# =======================================================
# =========== BIAS - UTILS FUNCTIONS -- START ===========
# =======================================================

def bin_age(age):
    for label, (low, high) in AGE_BINS.items():
        if low <= age <= high:
            return label
    return 'Unknown' 


def get_num_features(data):
    embed_cols = [col for col in data.columns if col.startswith('embed_')]
    if not embed_cols:
        raise ValueError('No "embed_" columns found in the DataFrame.')
    max_embed_num = max(int(col.split('_')[1]) for col in embed_cols)
    return max_embed_num


def stratified_sample_by_race(
    df, 
    n_samples=1000, 
    random_seed=42, 
    strata_columns=["sex", "disease", "age_bin"], 
    race_column="race", 
    output_dir=None, 
    csv_filename=None, 
    save_sample=False
):
    # Set random seed
    torch.manual_seed(random_seed)

    ## Step 1: Create composite strata
    df = df.copy()
    df["composite_strata"] = df[strata_columns].astype(str).agg("-".join, axis=1)

    ## Step 2: Calculate average proportions for composite strata across all races
    # Get proportions for each race
    race_groups = [df[df[race_column] == race] for race in df[race_column].unique()]
    all_labels = set(df["composite_strata"].unique())
    race_proportions = [
        group["composite_strata"].value_counts(normalize=True).reindex(all_labels, fill_value=0)
        for group in race_groups
    ]
    # Compute average proportions across races
    average_proportions = sum(race_proportions) / len(race_proportions)
    average_proportions_dict = average_proportions.to_dict()

    ## Step 3: Sample from each race while preserving average proportions
    balanced_race_groups = []
    for race in df[race_column].unique():
        race_group = df[df[race_column] == race]

        # Calculate weights for sampling
        observed_proportions = race_group["composite_strata"].value_counts(normalize=True).reindex(all_labels, fill_value=0)
        weights = {
            stratum: (average_proportions_dict.get(stratum, 0) / observed_proportions.get(stratum, 1))
            if observed_proportions.get(stratum, 1) > 0 else 0
            for stratum in all_labels
        }

        # Perform sampling
        ids = list(torch.utils.data.WeightedRandomSampler(
            race_group["composite_strata"].apply(lambda x: weights.get(x, 0)).values,
            n_samples,
            replacement=False  # Avoid duplicates
        ))
        balanced_race_groups.append(race_group.iloc[ids])

    ## Step 4: Combine sampled groups
    sampled_df = pd.concat(balanced_race_groups)

    ## Optionally save the sampled DataFrame
    if output_dir and csv_filename and save_sample:
        os.makedirs(output_dir, exist_ok=True)
        output_filepath = os.path.join(output_dir, csv_filename)
        sampled_df.to_csv(output_filepath)
        print(f"Stratified sampled data saved to: {output_filepath}")

    return sampled_df


def generate_multiple_samples(df, n_samples, n_iterations, strata_columns, race_column):
    samples_dfs = []
    print(f"Generating {n_iterations} samples, each sampled by race...")
    for i in range(1, n_iterations+1):
        # Using a different seed for each iteration to ensure variability in the samples
        sample_df = stratified_sample_by_race(df=df, n_samples=n_samples, random_seed=i, strata_columns=strata_columns, race_column=race_column)
        samples_dfs.append(sample_df)
    return samples_dfs


def verify_bias_data(df, expected_type):
    """Utility function to verify the data type of DataFrame columns."""
    if expected_type == 'binary':
        if not (df.select_dtypes(include=['bool']).shape[1] == df.shape[1] and df.isin([True, False]).all().all()):
            raise ValueError("All entries must be binary (True/False) for binary_rejections method.")
    elif expected_type == 'p-value':
        if not (df.select_dtypes(include=['number']).shape[1] == df.shape[1] and df.apply(lambda x: (x >= 0) & (x <= 1)).all().all()):
            raise ValueError("All entries must be numeric and within the range [0, 1] for p-value based methods.")


def prepare_bias_data_for_csv(modes, exp_var, data, columns, test_type, model):
    """ Prepare the DataFrame from the provided data and metadata. """
    data_dict = {'Mode': modes, 'Explained Variance': exp_var}
    for i, label in enumerate(columns[2:]):
        data_dict[label] = data[:, i]
    data_df = pd.DataFrame(data_dict, columns=columns)
    data_df.insert(0, 'Statistical Test Applied', test_type)  # Insert 'Statistical Test Applied' at index 0
    data_df.insert(0, 'Model Shortname', model['shortname'])  # Now insert 'Model Shortname' at index 0, pushing 'Statistical Test Applied' to index 1
    data_df.insert(0, 'Model Fullname', model['fullname'])  # Now insert 'Model Fullname' at index 0, pushing 'Model Shortname' to index 2
    return data_df


def save_bias_dataframe_to_csv(data_df, bias_stats_dir_path, test_type, filename):
    """ Save DataFrame to CSV in the specified directory. """
    # Define the directory for the specific statistical test type
    test_type_dir = os.path.join(bias_stats_dir_path, test_type)
    os.makedirs(test_type_dir, exist_ok=True)  # Ensure the directory exists
    file_csv_path = os.path.join(test_type_dir, filename)
    data_df.to_csv(file_csv_path, index=False)
    print(f"Data saved to: {file_csv_path}")


def setup_model_bias_analysis_directories(model_directory, mainsample_subdirectory_name='main_sample/',
                                          alluniquedata_subdirectory_name='all_unique_data_points/', simulation_subdirectory_name='simulation/'):
    """
    Creates directories within a single model's directory for storing different types of bias analysis outputs, 
    including specific subdirectories for detailed experiments.
    """
    # Base directory for bias analysis
    bias_dir_path = os.path.join(model_directory, 'bias_analysis2')

    # Define paths for various analysis aspects within the model's directory
    bias_stats_dir_path = os.path.join(bias_dir_path, 'bias_stats/')
    bias_stats_dir_path__main_sample = os.path.join(bias_stats_dir_path, mainsample_subdirectory_name)
    bias_stats_dir_path__all_unique_data_points = os.path.join(bias_stats_dir_path, alluniquedata_subdirectory_name)
    bias_stats_dir_path__simulation = os.path.join(bias_stats_dir_path, simulation_subdirectory_name)

    pca_dir_path = os.path.join(bias_dir_path, 'pca/')
    pca_plots_joint_dir_path = os.path.join(pca_dir_path, 'pca_plots_joint/')
    pca_plots_joint_dir_path__main_sample = os.path.join(pca_plots_joint_dir_path, mainsample_subdirectory_name)
    pca_plots_joint_dir_path__all_unique_data_points = os.path.join(pca_plots_joint_dir_path, alluniquedata_subdirectory_name)
    pca_plots_marginal_dir_path = os.path.join(pca_dir_path, 'pca_plots_marginal/')
    pca_plots_marginal_dir_path__main_sample = os.path.join(pca_plots_marginal_dir_path, mainsample_subdirectory_name)
    pca_plots_marginal_dir_path__all_unique_data_points = os.path.join(pca_plots_marginal_dir_path, alluniquedata_subdirectory_name)

    tsne_dir_path = os.path.join(bias_dir_path, 'tsne/')
    tsne_plots_joint_dir_path = os.path.join(tsne_dir_path, 'tsne_plots_joint/')
    tsne_plots_joint_dir_path__main_sample = os.path.join(tsne_plots_joint_dir_path, mainsample_subdirectory_name)
    tsne_plots_joint_dir_path__all_unique_data_points = os.path.join(tsne_plots_joint_dir_path, alluniquedata_subdirectory_name)
    tsne_plots_marginal_dir_path = os.path.join(tsne_dir_path, 'tsne_plots_marginal/')
    tsne_plots_marginal_dir_path__main_sample = os.path.join(tsne_plots_marginal_dir_path, mainsample_subdirectory_name)
    tsne_plots_marginal_dir_path__all_unique_data_points = os.path.join(tsne_plots_marginal_dir_path, alluniquedata_subdirectory_name)

    combined_plots_dir_path = os.path.join(bias_dir_path, 'combined_plots/')
    combined_joint_plots_dir_path = os.path.join(combined_plots_dir_path, 'combined_joint_plots/')
    combined_joint_plots_dir_path__main_sample = os.path.join(combined_joint_plots_dir_path, mainsample_subdirectory_name)
    combined_joint_plots_dir_path__all_unique_data_points = os.path.join(combined_joint_plots_dir_path, alluniquedata_subdirectory_name)
    combined_marginal_plots_dir_path = os.path.join(combined_plots_dir_path, 'combined_marginal_plots/')
    combined_marginal_plots_dir_path__main_sample = os.path.join(combined_marginal_plots_dir_path, mainsample_subdirectory_name)
    combined_marginal_plots_dir_path__all_unique_data_points = os.path.join(combined_marginal_plots_dir_path, alluniquedata_subdirectory_name)


    # Ensure the creation of these directories
    os.makedirs(bias_dir_path, exist_ok=True)

    os.makedirs(bias_stats_dir_path, exist_ok=True)
    os.makedirs(bias_stats_dir_path__main_sample, exist_ok=True)
    os.makedirs(bias_stats_dir_path__all_unique_data_points, exist_ok=True)
    os.makedirs(bias_stats_dir_path__simulation, exist_ok=True)

    os.makedirs(pca_dir_path, exist_ok=True)
    os.makedirs(pca_plots_joint_dir_path, exist_ok=True)
    os.makedirs(pca_plots_joint_dir_path__main_sample, exist_ok=True)
    os.makedirs(pca_plots_joint_dir_path__all_unique_data_points, exist_ok=True)
    os.makedirs(pca_plots_marginal_dir_path, exist_ok=True)
    os.makedirs(pca_plots_marginal_dir_path__main_sample, exist_ok=True)
    os.makedirs(pca_plots_marginal_dir_path__all_unique_data_points, exist_ok=True)

    os.makedirs(tsne_dir_path, exist_ok=True)
    os.makedirs(tsne_plots_joint_dir_path, exist_ok=True)
    os.makedirs(tsne_plots_joint_dir_path__main_sample, exist_ok=True)
    os.makedirs(tsne_plots_joint_dir_path__all_unique_data_points, exist_ok=True)
    os.makedirs(tsne_plots_marginal_dir_path, exist_ok=True)
    os.makedirs(tsne_plots_marginal_dir_path__main_sample, exist_ok=True)
    os.makedirs(tsne_plots_marginal_dir_path__all_unique_data_points, exist_ok=True)

    os.makedirs(combined_plots_dir_path, exist_ok=True)
    os.makedirs(combined_joint_plots_dir_path, exist_ok=True)
    os.makedirs(combined_joint_plots_dir_path__main_sample, exist_ok=True)
    os.makedirs(combined_joint_plots_dir_path__all_unique_data_points, exist_ok=True)
    os.makedirs(combined_marginal_plots_dir_path, exist_ok=True)
    os.makedirs(combined_marginal_plots_dir_path__main_sample, exist_ok=True)
    os.makedirs(combined_marginal_plots_dir_path__all_unique_data_points, exist_ok=True)


    return {
        'bias_dir_path': bias_dir_path,
        'bias_stats_dir_path': bias_stats_dir_path,
        'bias_stats_dir_path__main_sample': bias_stats_dir_path__main_sample,
        'bias_stats_dir_path__all_unique_data_points': bias_stats_dir_path__all_unique_data_points,
        'bias_stats_dir_path__simulation': bias_stats_dir_path__simulation,
        'pca_dir_path': pca_dir_path,
        'pca_plots_joint_dir_path': pca_plots_joint_dir_path,
        'pca_plots_joint_dir_path__main_sample': pca_plots_joint_dir_path__main_sample,
        'pca_plots_joint_dir_path__all_unique_data_points': pca_plots_joint_dir_path__all_unique_data_points,
        'pca_plots_marginal_dir_path': pca_plots_marginal_dir_path,
        'pca_plots_marginal_dir_path__main_sample': pca_plots_marginal_dir_path__main_sample,
        'pca_plots_marginal_dir_path__all_unique_data_points': pca_plots_marginal_dir_path__all_unique_data_points,
        'tsne_dir_path': tsne_dir_path,
        'tsne_plots_joint_dir_path': tsne_plots_joint_dir_path,
        'tsne_plots_joint_dir_path__main_sample': tsne_plots_joint_dir_path__main_sample,
        'tsne_plots_joint_dir_path__all_unique_data_points': tsne_plots_joint_dir_path__all_unique_data_points,
        'tsne_plots_marginal_dir_path': tsne_plots_marginal_dir_path,
        'tsne_plots_marginal_dir_path__main_sample': tsne_plots_marginal_dir_path__main_sample,
        'tsne_plots_marginal_dir_path__all_unique_data_points': tsne_plots_marginal_dir_path__all_unique_data_points,
        'combined_plots_dir_path': combined_plots_dir_path,
        'combined_joint_plots_dir_path': combined_joint_plots_dir_path,
        'combined_joint_plots_dir_path__main_sample': combined_joint_plots_dir_path__main_sample,
        'combined_joint_plots_dir_path__all_unique_data_points': combined_joint_plots_dir_path__all_unique_data_points,
        'combined_marginal_plots_dir_path': combined_marginal_plots_dir_path,
        'combined_marginal_plots_dir_path__main_sample': combined_marginal_plots_dir_path__main_sample,
        'combined_marginal_plots_dir_path__all_unique_data_points': combined_marginal_plots_dir_path__all_unique_data_points
    }

# =======================================================
# ============ BIAS - UTILS FUNCTIONS -- END ============
# =======================================================




# =======================================================
# ========= BIAS - INSPECTION FUNCTIONS - START =========
# =======================================================

def apply_pca(embeds, df, pca_dir_path, n_components=0.99):
    pca = PCA(n_components=n_components, whiten=False, random_state=RANDOM_SEED)
    embeds_pca = pca.fit_transform(embeds)

    # Extracting and storing PCA results
    df['PCA Mode 1'] = embeds_pca[:, 0]
    df['PCA Mode 2'] = embeds_pca[:, 1]
    df['PCA Mode 3'] = embeds_pca[:, 2]
    df['PCA Mode 4'] = embeds_pca[:, 3]

    # Calculate retained variance
    exp_var = pca.explained_variance_ratio_
    cumul_exp_var = np.cumsum(exp_var)

    # Plotting the explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(exp_var) + 1), cumul_exp_var, color='mediumblue')
    plt.xlabel('Mode', fontsize=12)
    plt.ylabel('Retained Variance', fontsize=12)
    plt.title('PCA Cumulative Explained Variance')
    
    # Dynamic tick marks: Calculate the interval to achieve 10 ticks including 1
    tick_interval = max(1, len(cumul_exp_var) // 10)
    tick_interval = int(round(tick_interval / 10) * 10)  # Adjusted to the nearest multiple of 10
    tick_interval = max(10, tick_interval)  # Ensure it's at least 10 or the calculated value
    ticks = [1] + list(range(tick_interval, len(cumul_exp_var) + 1, tick_interval))
    if ticks[-1] != len(cumul_exp_var):  # Ensure the last tick marks the last mode
        if len(cumul_exp_var) - ticks[-1] < 0.5 * tick_interval:  # Check if the last two ticks are too close
            ticks.pop()  # Remove the second last tick if too close
        ticks.append(len(cumul_exp_var))  # Append the last tick
    plt.xticks(ticks)
    
    # Save plot
    plot_path = os.path.join(pca_dir_path, 'pca_cumulative_explained_variance.png')
    plt.savefig(plot_path, dpi=OUT_DPI)
    plt.close()

    # Save explained variance data to CSV
    exp_var_df = pd.DataFrame({
        'PCA Mode': range(1, len(exp_var) + 1),
        'Explained Variance': exp_var,
        'Cumulative Explained Variance': cumul_exp_var
    })
    csv_path = os.path.join(pca_dir_path, 'pca_explained_variance.csv')
    exp_var_df.to_csv(csv_path, index=False)

    print(f"PCA output shape: {embeds_pca.shape}")
    print(f"Plot saved to: {plot_path}")
    print(f"Explained variance CSV saved to: {csv_path}")
    return embeds_pca, exp_var


def apply_tsne(embeds_pca, df, tsne_dir_path, n_components=2):
    # Apply t-SNE to the PCA-reduced embeddings
    tsne = TSNE(n_components=n_components, init='random', learning_rate='auto', random_state=RANDOM_SEED)
    embeds_tsne = tsne.fit_transform(embeds_pca)

    # Extracting and storing t-SNE results
    df['t-SNE Dimension 1'] = embeds_tsne[:, 0]
    df['t-SNE Dimension 2'] = embeds_tsne[:, 1]

    # Save the results to a CSV for further analysis
    tsne_df = pd.DataFrame({
        't-SNE Dimension 1': embeds_tsne[:, 0],
        't-SNE Dimension 2': embeds_tsne[:, 1]
    })
    csv_path = os.path.join(tsne_dir_path, 'tsne_results.csv')
    tsne_df.to_csv(csv_path, index=False)
    
    print(f"t-SNE data saved to: {csv_path}")
    return embeds_tsne


def perform_bias_inspection(sample_df, labels, pca_plots_joint_dir_path, pca_plots_marginal_dir_path, tsne_plots_joint_dir_path, tsne_plots_marginal_dir_path, 
                            std_centred, bias_dir_path, analysis_subdir_name, model, combined_joint_plots_dir_path, global_combined_joint_plots_dir_path,
                            combined_marginal_plots_dir_path, global_combined_marginal_plots_dir_path, dataset_name):
    """
    Perform bias analysis including plotting and statistical tests for a given sample of data.
    """
    # Define labels to guide the plotting
    labels_dict = {
        'Disease': {'hue_order': labels},
        'Sex': {'hue_order': [sex for sex in SEXES]},
        'Race': {'hue_order': [race for race in RACES]},
        'Age': {'hue_order': list(AGE_BINS.keys())}
    }

    ## Individual Plotting of PCA and t-SNE features
    plotting_methods = [
        ('PCA-1+2', [1, 2], 'PCA Mode 1', 'PCA Mode 2', pca_plots_joint_dir_path, pca_plots_marginal_dir_path),
        ('PCA-3+4', [3, 4], 'PCA Mode 3', 'PCA Mode 4', pca_plots_joint_dir_path, pca_plots_marginal_dir_path),
        ('tSNE', [1, 2], 't-SNE Dimension 1', 't-SNE Dimension 2', tsne_plots_joint_dir_path, tsne_plots_marginal_dir_path)
    ]
    for method, mode_indices, xdat, ydat, plots_joint_dir_path, plots_marginal_dir_path in plotting_methods:
        plot_feature_modes(df=sample_df, method=method, mode_indices=mode_indices, xdat=xdat, ydat=ydat, labels_dict=labels_dict, 
                           plots_joint_dir_path=plots_joint_dir_path, plots_marginal_dir_path=plots_marginal_dir_path, 
                           out_format=OUT_FORMAT, out_dpi=OUT_DPI, color_palette1=COLOR_PALETTE1, color_palette2=COLOR_PALETTE2,  
                           font_scale=FONT_SCALE, alpha=ALPHA, marker=MARKER, markersize=MARKERSIZE, kind=KIND, rasterized=RASTERIZED_SCATTER,
                           std_scale=STD_SCALE, std_centred=std_centred)

    ## Combined plots
    # Combine joint PCA and t-SNE plots together
    aggregate_jointplots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, model=model, 
                                   combined_joint_plots_dir_path=combined_joint_plots_dir_path,
                                   global_combined_joint_plots_dir_path=global_combined_joint_plots_dir_path,
                                   figsize_factor=5, plots_spacing=0, labels_fontsize=19.5, title_fontsize=22, dataset_name=dataset_name)
    # Combine marginal PCA and t-SNE plots together
    aggregate_marginalplots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, model=model, 
                                      combined_marginal_plots_dir_path=combined_marginal_plots_dir_path,
                                      global_combined_marginal_plots_dir_path=global_combined_marginal_plots_dir_path,
                                      figsize_factor=5, plots_spacing=0, labels_fontsize=15.5, title_fontsize=18, dataset_name=dataset_name)

# =======================================================
# ========== BIAS - INSPECTION FUNCTIONS - END ==========
# =======================================================




# =======================================================
# ==== BIAS - INSPECTION PLOTTING FUNCTIONS -- START ====
# =======================================================

def plot_feature_modes(df, method, mode_indices, xdat, ydat, labels_dict, plots_joint_dir_path, plots_marginal_dir_path, 
                       out_format, out_dpi, color_palette1, color_palette2, font_scale, alpha, marker, markersize, kind, rasterized, 
                       std_scale, std_centred=False):
    sns.set_theme(style="white", palette=color_palette1, font_scale=font_scale, font='Latin Modern Roman')
    xlim = None
    ylim = None
    
    # Loop through each label in labels_dict to create plots for each category
    for i, (label, settings) in enumerate(labels_dict.items()):
        sns.set_theme(style="white", font_scale=font_scale, font='Latin Modern Roman')
        current_palette = color_palette2 if label == 'Age' else color_palette1
        # Jointplot for each label
        if i == 0:
            if std_centred:
                # Calculate dynamic limits for the plots based on the data standard deviation
                xlim = [-std_scale * np.std(df[xdat]), std_scale * np.std(df[xdat])]
                ylim = [-std_scale * np.std(df[ydat]), std_scale * np.std(df[ydat])]
                # Apply xlim and ylim
                fig = sns.jointplot(x=xdat, y=ydat, hue=label, kind=kind, alpha=alpha, marker=marker, s=markersize,
                                    palette=current_palette, hue_order=settings['hue_order'], data=df, 
                                    joint_kws={'rasterized': rasterized}, marginal_kws={'common_norm': False}, 
                                    xlim=xlim, ylim=ylim)
            else:    
                fig = sns.jointplot(x=xdat, y=ydat, hue=label, kind=kind, alpha=alpha, marker=marker, s=markersize,
                                    palette=current_palette, hue_order=settings['hue_order'], data=df, 
                                    joint_kws={'rasterized': rasterized}, marginal_kws={'common_norm': False})
                # For the first plot, fetch axis limits to use in subsequent plots (if std_centred has not been set to True)
                xlim = fig.ax_joint.get_xlim()
                ylim = fig.ax_joint.get_ylim()
        else:
            # Apply previously determined xlim and ylim
            fig = sns.jointplot(x=xdat, y=ydat, hue=label, kind=kind, alpha=alpha, marker=marker, s=markersize,
                                palette=current_palette, hue_order=settings['hue_order'], data=df, 
                                joint_kws={'rasterized': rasterized}, marginal_kws={'common_norm': False}, 
                                xlim=xlim, ylim=ylim)
            
        fig.ax_joint.set_xlabel(xdat, fontsize=18)
        fig.ax_joint.set_ylabel(ydat, fontsize=18)
        fig.ax_joint.legend(loc='upper right', fontsize=16)
        joint_filename = f'{method}-{label}-joint.{out_format}'
        joint_filepath = os.path.join(plots_joint_dir_path, joint_filename)
        plt.savefig(joint_filepath, bbox_inches='tight', dpi=out_dpi)
        plt.close()
        
        # Marginal KDE plots for x and y
        for axdat, index, axlim in zip([xdat, ydat], mode_indices, [xlim, ylim]):
            fig, ax = plt.subplots(figsize=(10, 3))
            p = sns.kdeplot(x=axdat, hue=label, fill=True, hue_order=settings['hue_order'], data=df, ax=ax, 
                            palette=current_palette, common_norm=False)
            p.get_legend().set_title(None)
            p.spines[['right', 'top']].set_visible(False)
            p.set_xlim(axlim)
            ax.set_xlabel(axdat, fontsize=22)
            ax.set_ylabel('Density', fontsize=22)
            
            method_name = method.split('-')[0]
            marginal_filename = f'{method_name}-{index}-{label}-marginal.{out_format}'
            marginal_filepath = os.path.join(plots_marginal_dir_path, marginal_filename)
            plt.savefig(marginal_filepath, bbox_inches='tight', dpi=out_dpi)
            plt.close()

        # Marginal KDE plots for x and y with scaled-up legend
        sns.set_theme(style="white", font_scale=font_scale + 0.35, font='Latin Modern Roman')
        for axdat, index, axlim in zip([xdat, ydat], mode_indices, [xlim, ylim]):
            fig, ax = plt.subplots(figsize=(10, 3))
            p = sns.kdeplot(x=axdat, hue=label, fill=True, hue_order=settings['hue_order'], data=df, ax=ax, 
                            palette=current_palette, common_norm=False)
            p.get_legend().set_title(None)
            p.spines[['right', 'top']].set_visible(False)
            p.set_xlim(axlim)
            ax.set_xlabel(axdat, fontsize=22)
            ax.set_ylabel('Density', fontsize=22)
            
            method_name = method.split('-')[0]
            marginal_filename = f'{method_name}-{index}-{label}-ScaledUpLegend-marginal.{out_format}'
            marginal_filepath = os.path.join(plots_marginal_dir_path, marginal_filename)
            plt.savefig(marginal_filepath, bbox_inches='tight', dpi=out_dpi)
            plt.close()

        print(f"Plots for {label} saved to: (Joint) {joint_filepath} and (Marginal) {marginal_filepath}")


def aggregate_plots_into_grid(bias_dir_path, analysis_subdir_name, combined_plots_dir_path, plot_files, grid_shape, cols_names, title, 
                              combined_plots_grid_filename, figsize_factor=5, plots_spacing=0, labels_fontsize=13.5, title_fontsize=16, 
                              joint=True, pca_and_tsne=True, include_age=True, global_save_dir_path=None):
    # Adjust the grid shape and columns names if "Age" is not included
    if not include_age:
        grid_shape = (grid_shape[0], grid_shape[1] - 1)  # Reduce the number of columns by one
        cols_names = cols_names[:-1]  # Remove the last column name, assuming it is "Age"
        plot_files = [pf for pf in plot_files if pf[2] != 3]  # Exclude plot files where column index is 3 (Age)

    title_height_ratio = 0.2 if joint else 0.5
    if not joint and pca_and_tsne:
        height_ratios = [title_height_ratio] + [1]*4 + [0.25] + [1]*(grid_shape[0]-4)  # Increased spacing for first t-SNE row to contrast with PCA above
    else:
        height_ratios = [title_height_ratio] + [1]*grid_shape[0]

    height_unadjusted = sum(height_ratios) * figsize_factor
    height = height_unadjusted if joint else height_unadjusted/2.5
    width = grid_shape[1] * figsize_factor
    fig = plt.figure(figsize=(width, height))

    grid_shape_0 = grid_shape[0]+2 if not joint and pca_and_tsne else grid_shape[0]+1
    gs = gridspec.GridSpec(grid_shape_0, grid_shape[1], figure=fig, height_ratios=height_ratios, wspace=plots_spacing, hspace=plots_spacing)

    # Load and display each plot
    for filename, row, col in plot_files:
        dim_red_technique_dirname = filename.split('-')[0].lower()
        dim_red_technique_plots_dirname = f'{dim_red_technique_dirname}_plots_joint' if joint else f'{dim_red_technique_dirname}_plots_marginal'
        if not joint and not include_age:  # for marginal plots without 'Age', use the scaled-up-legend plots
            filename_splits = filename.rsplit('-', 1)
            filename = f"{filename_splits[0]}-ScaledUpLegend-{filename_splits[1]}"
        path = os.path.join(bias_dir_path, dim_red_technique_dirname, dim_red_technique_plots_dirname, analysis_subdir_name, filename)
        img = mpimg.imread(path)
        # Adjust subplot creation based on t-SNE rows
        if not joint and pca_and_tsne and row >= 4:  # Assuming t-SNE plots start from row 5 (i.e., index 4)
            ax = fig.add_subplot(gs[row+2, col])
        else:
            ax = fig.add_subplot(gs[row+1, col])
        ax.imshow(img)
        ax.axis('off')

    # Set overall titles and axis labels as described
    ax_titles = [fig.add_subplot(gs[1, i], frameon=False) for i in range(grid_shape[1])]  # Create subplots for titles in the first row
    for ax, col in zip(ax_titles, cols_names):
        ax.set_title(col, fontweight='bold', fontsize=labels_fontsize, pad=10)
        ax.axis('off')

    if pca_and_tsne:
        # Creating a special y-axis label for PCA that spans the first two rows
        # This creates an invisible axes and sets a centered label
        ax_pca_label = fig.add_subplot(gs[0+1:2+1, 0] if joint else gs[0+1:4+1, 0], frameon=False)
        ax_pca_label.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax_pca_label.set_ylabel('PCA', size=labels_fontsize, rotation='vertical', labelpad=0, verticalalignment='center', fontweight='bold')
        ax_pca_label.grid(False)

        # Set y-axis label for t-SNE in the third row
        ax_tsne_label = fig.add_subplot(gs[2+1, 0] if joint else gs[4+2:6+2, 0], frameon=False)
        ax_tsne_label.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        ax_tsne_label.set_ylabel('t-SNE', size=labels_fontsize, rotation='vertical', labelpad=0, verticalalignment='center', fontweight='bold')
        ax_tsne_label.grid(False)

    # Set an overall title
    ax_title = fig.add_subplot(gs[0, :], frameon=False)  # Span all columns in the first row
    ax_title.set_title(title, fontsize=title_fontsize, pad=10)
    ax_title.axis('off')

    # Save the combined figure
    jointplots_grid_path = os.path.join(combined_plots_dir_path, combined_plots_grid_filename)
    plt.savefig(jointplots_grid_path, dpi=OUT_DPI)
    if global_save_dir_path:
        global_jointplots_grid_path = os.path.join(global_save_dir_path, combined_plots_grid_filename)
        plt.savefig(global_jointplots_grid_path, dpi=OUT_DPI)
    plt.close()
    print(f"Combined figure '{title}' saved to: {jointplots_grid_path}")


def aggregate_jointplots_into_grid(bias_dir_path, analysis_subdir_name, model, combined_joint_plots_dir_path, global_combined_joint_plots_dir_path, 
                                   figsize_factor=4, plots_spacing=0, labels_fontsize=13.5, title_fontsize=16, dataset_name='CheXpert'):
    """
    Aggregates PCA and t-SNE joint plots into a unified grid for comprehensive data visualisation.
    Also, plots corresponding unified grids for PCA and t-SNE separately.
    """
    # Define configurations for full grid with PCA and t-SNE joint plots
    all_joint_plot_files = [
        ("PCA-1+2-Disease-joint.png", 0, 0),
        ("PCA-1+2-Sex-joint.png", 0, 1),
        ("PCA-1+2-Race-joint.png", 0, 2),
        ("PCA-1+2-Age-joint.png", 0, 3),
        ("PCA-3+4-Disease-joint.png", 1, 0),
        ("PCA-3+4-Sex-joint.png", 1, 1),
        ("PCA-3+4-Race-joint.png", 1, 2),
        ("PCA-3+4-Age-joint.png", 1, 3),
        ("tSNE-Disease-joint.png", 2, 0),
        ("tSNE-Sex-joint.png", 2, 1),
        ("tSNE-Race-joint.png", 2, 2),
        ("tSNE-Age-joint.png", 2, 3)
    ]
    cols_names = ['Disease', 'Sex', 'Race', 'Age']
    title = f"{model['shortname']}\nPCA and t-SNE Joint Plots | {dataset_name}"
    jointplots_grid_filename = f"{model['shortname']}__PCAandtSNE_Joint_Plots__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_joint_plots_dir_path, 
                              plot_files=all_joint_plot_files, grid_shape=(3, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=jointplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=True, pca_and_tsne=True, include_age=True, global_save_dir_path=global_combined_joint_plots_dir_path)  # Save output to global models dir
    jointplots_grid_filename = f"{model['shortname']}__PCAandtSNE_Joint_Plots_withoutAge__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_joint_plots_dir_path, 
                              plot_files=all_joint_plot_files, grid_shape=(3, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=jointplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=True, pca_and_tsne=True, include_age=False)
    
    # Separate configurations for PCA-only joint plots
    pca_joint_plot_files = [
        ("PCA-1+2-Disease-joint.png", 0, 0),
        ("PCA-1+2-Sex-joint.png", 0, 1),
        ("PCA-1+2-Race-joint.png", 0, 2),
        ("PCA-1+2-Age-joint.png", 0, 3),
        ("PCA-3+4-Disease-joint.png", 1, 0),
        ("PCA-3+4-Sex-joint.png", 1, 1),
        ("PCA-3+4-Race-joint.png", 1, 2),
        ("PCA-3+4-Age-joint.png", 1, 3)
    ]
    cols_names = ['Disease', 'Sex', 'Race', 'Age']
    title = f"{model['shortname']}\nPCA Joint Plots | {dataset_name}"
    jointplots_grid_filename = f"{model['shortname']}__PCA_Joint_Plots__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_joint_plots_dir_path, 
                              plot_files=pca_joint_plot_files, grid_shape=(2, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=jointplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=True, pca_and_tsne=False, include_age=True)
    jointplots_grid_filename = f"{model['shortname']}__PCA_Joint_Plots_withoutAge__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_joint_plots_dir_path, 
                              plot_files=pca_joint_plot_files, grid_shape=(2, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=jointplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=True, pca_and_tsne=False, include_age=False)
    
    # Separate configurations for t-SNE-only joint plots
    tsne_joint_plot_files = [
        ("tSNE-Disease-joint.png", 0, 0),
        ("tSNE-Sex-joint.png", 0, 1),
        ("tSNE-Race-joint.png", 0, 2),
        ("tSNE-Age-joint.png", 0, 3)
    ]
    cols_names = ['Disease', 'Sex', 'Race', 'Age']
    title = f"{model['shortname']}\nt-SNE Joint Plots | {dataset_name}"
    jointplots_grid_filename = f"{model['shortname']}__tSNE_Joint_Plots__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_joint_plots_dir_path, 
                              plot_files=tsne_joint_plot_files, grid_shape=(1, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=jointplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=True, pca_and_tsne=False, include_age=True)
    jointplots_grid_filename = f"{model['shortname']}__tSNE_Joint_Plots_withoutAge__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_joint_plots_dir_path, 
                              plot_files=tsne_joint_plot_files, grid_shape=(1, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=jointplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=True, pca_and_tsne=False, include_age=False)
    

def aggregate_marginalplots_into_grid(bias_dir_path, analysis_subdir_name, model, combined_marginal_plots_dir_path, global_combined_marginal_plots_dir_path,
                                      figsize_factor=4, plots_spacing=0, labels_fontsize=13.5, title_fontsize=16, dataset_name='CheXpert'):
    """
    Aggregates PCA and t-SNE marginal plots into a unified grid for comprehensive data visualisation.
    Also, plots corresponding unified grids for PCA and t-SNE separately.
    """
    # Define configurations for full grid with PCA and t-SNE marginal plots
    all_joint_plot_files = [
        ("PCA-1-Disease-marginal.png", 0, 0),
        ("PCA-1-Sex-marginal.png", 0, 1),
        ("PCA-1-Race-marginal.png", 0, 2),
        ("PCA-1-Age-marginal.png", 0, 3),
        ("PCA-2-Disease-marginal.png", 1, 0),
        ("PCA-2-Sex-marginal.png", 1, 1),
        ("PCA-2-Race-marginal.png", 1, 2),
        ("PCA-2-Age-marginal.png", 1, 3),
        ("PCA-3-Disease-marginal.png", 2, 0),
        ("PCA-3-Sex-marginal.png", 2, 1),
        ("PCA-3-Race-marginal.png", 2, 2),
        ("PCA-3-Age-marginal.png", 2, 3),
        ("PCA-4-Disease-marginal.png", 3, 0),
        ("PCA-4-Sex-marginal.png", 3, 1),
        ("PCA-4-Race-marginal.png", 3, 2),
        ("PCA-4-Age-marginal.png", 3, 3),
        ("tSNE-1-Disease-marginal.png", 4, 0),
        ("tSNE-1-Sex-marginal.png", 4, 1),
        ("tSNE-1-Race-marginal.png", 4, 2),
        ("tSNE-1-Age-marginal.png", 4, 3),
        ("tSNE-2-Disease-marginal.png", 5, 0),
        ("tSNE-2-Sex-marginal.png", 5, 1),
        ("tSNE-2-Race-marginal.png", 5, 2),
        ("tSNE-2-Age-marginal.png", 5, 3)
    ]
    cols_names = ['Disease', 'Sex', 'Race', 'Age']
    title = f"{model['shortname']}\nPCA and t-SNE Marginal Plots | {dataset_name}"
    marginalplots_grid_filename = f"{model['shortname']}__PCAandtSNE_Marginal_Plots__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_marginal_plots_dir_path, 
                              plot_files=all_joint_plot_files, grid_shape=(6, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=marginalplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=False, pca_and_tsne=True, include_age=True)
    marginalplots_grid_filename = f"{model['shortname']}__PCAandtSNE_Marginal_Plots_withoutAge__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_marginal_plots_dir_path, 
                              plot_files=all_joint_plot_files, grid_shape=(6, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=marginalplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=False, pca_and_tsne=True, include_age=False, global_save_dir_path=global_combined_marginal_plots_dir_path)  # Save output to global models dir
    
    # Separate configurations for PCA-only marginal plots
    pca_joint_plot_files = [
        ("PCA-1-Disease-marginal.png", 0, 0),
        ("PCA-1-Sex-marginal.png", 0, 1),
        ("PCA-1-Race-marginal.png", 0, 2),
        ("PCA-1-Age-marginal.png", 0, 3),
        ("PCA-2-Disease-marginal.png", 1, 0),
        ("PCA-2-Sex-marginal.png", 1, 1),
        ("PCA-2-Race-marginal.png", 1, 2),
        ("PCA-2-Age-marginal.png", 1, 3),
        ("PCA-3-Disease-marginal.png", 2, 0),
        ("PCA-3-Sex-marginal.png", 2, 1),
        ("PCA-3-Race-marginal.png", 2, 2),
        ("PCA-3-Age-marginal.png", 2, 3),
        ("PCA-4-Disease-marginal.png", 3, 0),
        ("PCA-4-Sex-marginal.png", 3, 1),
        ("PCA-4-Race-marginal.png", 3, 2),
        ("PCA-4-Age-marginal.png", 3, 3)
    ]
    cols_names = ['Disease', 'Sex', 'Race', 'Age']
    title = f"{model['shortname']}\nPCA Marginal Plots | {dataset_name}"
    marginalplots_grid_filename = f"{model['shortname']}__PCA_Marginal_Plots__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_marginal_plots_dir_path, 
                              plot_files=pca_joint_plot_files, grid_shape=(4, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=marginalplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=False, pca_and_tsne=False, include_age=True)
    marginalplots_grid_filename = f"{model['shortname']}__PCA_Marginal_Plots_withoutAge__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_marginal_plots_dir_path, 
                              plot_files=pca_joint_plot_files, grid_shape=(4, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=marginalplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=False, pca_and_tsne=False, include_age=False)
    
    # Separate configurations for t-SNE-only marginal plots
    tsne_joint_plot_files = [
        ("tSNE-1-Disease-marginal.png", 0, 0),
        ("tSNE-1-Sex-marginal.png", 0, 1),
        ("tSNE-1-Race-marginal.png", 0, 2),
        ("tSNE-1-Age-marginal.png", 0, 3),
        ("tSNE-2-Disease-marginal.png", 1, 0),
        ("tSNE-2-Sex-marginal.png", 1, 1),
        ("tSNE-2-Race-marginal.png", 1, 2),
        ("tSNE-2-Age-marginal.png", 1, 3)
    ]
    cols_names = ['Disease', 'Sex', 'Race', 'Age']
    title = f"{model['shortname']}\nt-SNE Marginal Plots | {dataset_name}"
    marginalplots_grid_filename = f"{model['shortname']}__tSNE_Marginal_Plots__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_marginal_plots_dir_path, 
                              plot_files=tsne_joint_plot_files, grid_shape=(2, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=marginalplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=False, pca_and_tsne=False, include_age=True)
    marginalplots_grid_filename = f"{model['shortname']}__tSNE_Marginal_Plots_withoutAge__({dataset_name}).png"
    aggregate_plots_into_grid(bias_dir_path=bias_dir_path, analysis_subdir_name=analysis_subdir_name, combined_plots_dir_path=combined_marginal_plots_dir_path, 
                              plot_files=tsne_joint_plot_files, grid_shape=(2, 4), cols_names=cols_names, title=title, 
                              combined_plots_grid_filename=marginalplots_grid_filename,  figsize_factor=figsize_factor, 
                              plots_spacing=plots_spacing, labels_fontsize=labels_fontsize, title_fontsize=title_fontsize, 
                              joint=False, pca_and_tsne=False, include_age=False)
    
# =======================================================
# ===== BIAS - INSPECTION PLOTTING FUNCTIONS -- END =====
# =======================================================




# =======================================================
# ========= BIAS - STATISTICAL ANALYSIS - START =========
# =======================================================

def perform_statistical_tests(sample_df, bias_stats_dir_path, races, sexes, diseases, exp_var, model, dataset_name='CheXpert', test_type='ks',
                              save_to_csv=True):
    # Helper function to perform the specified statistical test between two groups 
    # See: [Independent Sample Tests--from SciPy: https://docs.scipy.org/doc/scipy/reference/stats.html#independent-sample-tests]
    # (default to Two-sample Kolmogorov–Smirnov tests between each pair of subgroups)
    def stats_tests(marginal, samples, test_type):
        groups = {race: samples[samples['race'] == race] for race in races}
        groups.update({sex: samples[samples['sex'] == sex] for sex in sexes})
        groups.update({disease: samples[samples['disease'] == disease] for disease in diseases})
        # Define comparisons
        comparisons = [
            (diseases[0], diseases[1]), 
            (races[0], races[1]),  
            (races[0], races[2]),
            (races[1], races[2]),
            (sexes[0], sexes[1])
        ]
        # Perform Two-sample statistical tests
        results = []
        for group1, group2 in comparisons:
            data1 = groups[group1][marginal]
            data2 = groups[group2][marginal]

            if test_type == 'ks':
                result = ks_2samp(data1, data2).pvalue
            elif test_type == 'ks_permutation':
                result = run_ks_permutation_test(data1, data2, n_permutations=N_PERMUTATIONS)
            elif test_type == 'mannwhitney':
                result = mannwhitneyu(data1, data2).pvalue
            elif test_type == 'anderson':
                result = anderson_ksamp([data1, data2]).pvalue
            elif test_type == 'median':
                result = median_test(data1, data2).pvalue
            elif test_type == 'mood':
                result = mood(data1, data2).pvalue
            elif test_type == 'kruskal':
                result = kruskal(data1, data2).pvalue
            elif test_type == 'cramervonmises':
                result = cramervonmises_2samp(data1, data2).pvalue
            else:
                raise ValueError("Unsupported test type provided")
            
            results.append(result)
        return results

    # Get p-values for each PCA mode
    pvals = np.array([stats_tests(f'PCA Mode {i+1}', sample_df, test_type) for i in range(4)])
    
    # Adjust p-values using the FDR method (Benjamini-Yekutieli Procedure)
    res = multipletests(pvals.flatten(), alpha=0.05, method='fdr_by', is_sorted=False, returnsorted=False)
    reshaped_rejected = np.array(res[0]).reshape((4,5))
    reshaped_adjusted_pvals = np.array(res[1]).reshape((4,5))

    # Create a DataFrame for the results
    modes = [f'PCA Mode {i+1}' for i in range(4)]
    columns = ['Mode', 'Explained Variance', f'{diseases[0]} vs {diseases[1]}', f'{races[0]} vs {races[1]}', 
               f'{races[0]} vs {races[2]}', f'{races[1]} vs {races[2]}', f'{sexes[0]} vs {sexes[1]}']
    

    # Prepare and insert data for adjusted p-values (optionnaly save into csv)
    adjusted_pvalues_df = prepare_bias_data_for_csv(modes=modes, exp_var=exp_var[:4], data=reshaped_adjusted_pvals, columns=columns,
                                                    test_type=test_type, model=model)
    if save_to_csv:
        save_bias_dataframe_to_csv(data_df=adjusted_pvalues_df, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--statistical_tests_adjusted_pvalues__({dataset_name}).csv")

    # Prepare and insert data for rejection results (optionnaly save into csv)
    rejection_df = prepare_bias_data_for_csv(modes=modes, exp_var=exp_var[:4], data=reshaped_rejected, columns=columns,
                                             test_type=test_type, model=model)
    if save_to_csv:
        save_bias_dataframe_to_csv(data_df=rejection_df, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--statistical_tests_rejections__({dataset_name}).csv")
    
    return adjusted_pvalues_df, rejection_df


def run_ks_permutation_test(source, target, n_permutations=1000, random_seed=42):
    """
    Runs a permutation-based Kolmogorov-Smirnov test to compare two samples.
    
    Args:
        source (1D array-like): e.g., list, NumPy array, or pandas Series of feature values from the source set.
        target (1D array-like): e.g., list, NumPy array, or pandas Series of feature values from the target set.
        n_permutations (int): Number of permutations to perform.
    
    Returns:
        float: p-value from the permutation KS test.
    
    Description:
        This function computes the KS statistic for the actual data and then
        compares this to the distribution of KS statistics generated by randomly
        permuting the combined dataset. The p-value represents the proportion
        of the permuted datasets that resulted in a KS statistic as extreme
        or more extreme than the observed statistic.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n1, n2 = len(source), len(target)
    observed_ks_statistic = ks_2samp(source, target, method="exact").statistic
    
    # Combine and permute data
    combined_set = np.concatenate([source, target], axis=0)
    
    # Generate permuted statistics
    permuted_stats = []
    for _ in range(n_permutations):
        permuted_indices = np.random.permutation(n1 + n2)
        permuted_statistic = ks_2samp(combined_set[permuted_indices[:n1]], combined_set[permuted_indices[n1:]], method="exact").statistic
        permuted_stats.append(permuted_statistic)
    
    # Calculate p-value as the proportion of permuted stats greater than or equal to the observed
    p_value = (np.sum(np.array(permuted_stats) >= observed_ks_statistic) + 1) / (n_permutations + 1)  # +1 added to avoid strictly zero p-values and ensure continuity

    return p_value


def perform_bias_statistical_analysis(sample_df, bias_stats_dir_path, exp_var, model, dataset_name, stat_test_types, diseases=['Pleural Effusion', 'No Finding']):
    """Statistical Analysis: Perform Two-sample Independent tests between each pair of subgroups"""
    # Lists to store adjusted_pvalues and rejection DataFrames as well as corresponding expanded representations for the model for each test_type applied
    combined_pvalues_dfs = []
    combined_binary_rejections_dfs = [] 
    combined_categorised_rejections_dfs__detailed = []
    combined_categorised_rejections_dfs__compact = []
    combined_categorised2_rejections_dfs__detailed = []
    combined_categorised2_rejections_dfs__compact = []

    # Loop over each test type and perform statistical analysis
    for test_type in stat_test_types:
        print(f"Performing {test_type} test...")

        adjusted_pvalues_df, rejection_df = perform_statistical_tests(
            sample_df=sample_df,
            bias_stats_dir_path=bias_stats_dir_path,
            races=RACES,
            sexes=SEXES,
            diseases=diseases,
            exp_var=exp_var,
            model=model,
            dataset_name=dataset_name,
            test_type=test_type
        )

        # Combine results for this test type
        combined_categorised_rejections_df__detailed, combined_categorised_rejections_df__compact = combine_simulation_results([adjusted_pvalues_df], method='categorised_rejections')
        combined_categorised2_rejections_df__detailed, combined_categorised2_rejections_df__compact = combine_simulation_results([adjusted_pvalues_df], method='categorised2_rejections')

        # Save the combined results into csv files
        save_bias_dataframe_to_csv(data_df=combined_categorised_rejections_df__detailed, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_categorised_rejections--detailed__({dataset_name}).csv")
        save_bias_dataframe_to_csv(data_df=combined_categorised_rejections_df__compact, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_categorised_rejections--compact__({dataset_name}).csv")
        save_bias_dataframe_to_csv(data_df=combined_categorised2_rejections_df__detailed, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_categorised2_rejections--detailed__({dataset_name}).csv")
        save_bias_dataframe_to_csv(data_df=combined_categorised2_rejections_df__compact, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_categorised2_rejections--compact__({dataset_name}).csv")
        
        # Store combined results to list
        combined_pvalues_dfs.append(adjusted_pvalues_df)
        combined_binary_rejections_dfs.append(rejection_df)
        combined_categorised_rejections_dfs__detailed.append(combined_categorised_rejections_df__detailed)
        combined_categorised_rejections_dfs__compact.append(combined_categorised_rejections_df__compact)
        combined_categorised2_rejections_dfs__detailed.append(combined_categorised2_rejections_df__detailed)
        combined_categorised2_rejections_dfs__compact.append(combined_categorised2_rejections_df__compact)

    # Concatenate all combined DataFrames for p-values and rejections
    final_pvalues_df = pd.concat(combined_pvalues_dfs, ignore_index=True)
    final_binary_rejections_df = pd.concat(combined_binary_rejections_dfs, ignore_index=True)
    final_categorised_rejections_df__detailed = pd.concat(combined_categorised_rejections_dfs__detailed, ignore_index=True)
    final_categorised_rejections_df__compact = pd.concat(combined_categorised_rejections_dfs__compact, ignore_index=True)
    final_categorised2_rejections_df__detailed = pd.concat(combined_categorised2_rejections_dfs__detailed, ignore_index=True)
    final_categorised2_rejections_df__compact = pd.concat(combined_categorised2_rejections_dfs__compact, ignore_index=True)

    # Save these concatenated DataFrames to CSV
    final_pvalues_path = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--statistical_tests_adjusted_pvalues__({dataset_name}).csv")
    final_pvalues_df.to_csv(final_pvalues_path, index=False)
    final_binary_rejections_path = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--statistical_tests_binary_rejections__({dataset_name}).csv")
    final_binary_rejections_df.to_csv(final_binary_rejections_path, index=False)
    final_categorised_rejections_path__detailed = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--statistical_tests_categorised_rejections--detailed__({dataset_name}).csv")
    final_categorised_rejections_df__detailed.to_csv(final_categorised_rejections_path__detailed, index=False)
    final_categorised_rejections_path__compact = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--statistical_tests_categorised_rejections--compact__({dataset_name}).csv")
    final_categorised_rejections_df__compact.to_csv(final_categorised_rejections_path__compact, index=False)
    final_categorised2_rejections_path__detailed = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--statistical_tests_categorised2_rejections--detailed__({dataset_name}).csv")
    final_categorised2_rejections_df__detailed.to_csv(final_categorised2_rejections_path__detailed, index=False)
    final_categorised2_rejections_path__compact = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--statistical_tests_categorised2_rejections--compact__({dataset_name}).csv")
    final_categorised2_rejections_df__compact.to_csv(final_categorised2_rejections_path__compact, index=False)

# =======================================================
# ========== BIAS - STATISTICAL ANALYSIS - END ==========
# =======================================================




# =======================================================
# ====== BIAS - STATS SIMULATION FUNCTIONS - START ======
# =======================================================

def simulate_bias_statistical_analysis(samples_dfs, bias_stats_dir_path, exp_var, model, dataset_name, stat_test_types, diseases=['Pleural Effusion', 'No Finding']):
    # Initialise storage lists for results across all test types
    combined_pvalues_dfs__detailed = []
    combined_pvalues_dfs__compact = []
    combined_binary_rejections_dfs__detailed = []
    combined_binary_rejections_dfs__compact = []
    combined_categorised_rejections_dfs__detailed = []
    combined_categorised_rejections_dfs__compact = []
    combined_categorised2_rejections_dfs__detailed = []
    combined_categorised2_rejections_dfs__compact = []

    # Loop over each test type and perform statistical analysis
    for test_type in stat_test_types:
        print(f"Performing {test_type} test...")

        # Initialise lists to store results for the current test type
        all_pvalues = []
        all_rejections = []

        # Perform tests for each sample and store the results
        for sample_df in samples_dfs:
            adjusted_pvalues_df, rejection_df = perform_statistical_tests(
                sample_df=sample_df,
                bias_stats_dir_path=bias_stats_dir_path,
                races=RACES,
                sexes=SEXES,
                diseases=diseases,
                exp_var=exp_var,
                model=model,
                dataset_name=dataset_name,
                test_type=test_type,
                save_to_csv=False
            )
            all_pvalues.append(adjusted_pvalues_df)
            all_rejections.append(rejection_df)

        # Combine results for this test type
        combined_pvalues_df__detailed, combined_pvalues_df__compact = combine_simulation_results(all_pvalues, method='mean_std')
        combined_binary_rejections_df__detailed, combined_binary_rejections_df__compact = combine_simulation_results(all_rejections, method='binary_rejections')
        combined_categorised_rejections_df__detailed, combined_categorised_rejections_df__compact = combine_simulation_results(all_pvalues, method='categorised_rejections')
        combined_categorised2_rejections_df__detailed, combined_categorised2_rejections_df__compact = combine_simulation_results(all_pvalues, method='categorised2_rejections')

        # Save the combined results into csv files
        save_bias_dataframe_to_csv(data_df=combined_pvalues_df__detailed, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_adjusted_pvalues--detailed__({dataset_name}).csv")
        save_bias_dataframe_to_csv(data_df=combined_pvalues_df__compact, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_adjusted_pvalues--compact__({dataset_name}).csv")
        save_bias_dataframe_to_csv(data_df=combined_binary_rejections_df__detailed, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_binary_rejections--detailed__({dataset_name}).csv")
        save_bias_dataframe_to_csv(data_df=combined_binary_rejections_df__compact, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_binary_rejections--compact__({dataset_name}).csv")
        save_bias_dataframe_to_csv(data_df=combined_categorised_rejections_df__detailed, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_categorised_rejections--detailed__({dataset_name}).csv")
        save_bias_dataframe_to_csv(data_df=combined_categorised_rejections_df__compact, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_categorised_rejections--compact__({dataset_name}).csv")
        save_bias_dataframe_to_csv(data_df=combined_categorised2_rejections_df__detailed, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_categorised2_rejections--detailed__({dataset_name}).csv")
        save_bias_dataframe_to_csv(data_df=combined_categorised2_rejections_df__compact, bias_stats_dir_path=bias_stats_dir_path, test_type=test_type,
                                   filename=f"{model['shortname']}__{test_type}--combined_statistical_tests_categorised2_rejections--compact__({dataset_name}).csv")
        
        # Store combined results to list
        combined_pvalues_dfs__detailed.append(combined_pvalues_df__detailed)
        combined_pvalues_dfs__compact.append(combined_pvalues_df__compact)
        combined_binary_rejections_dfs__detailed.append(combined_binary_rejections_df__detailed)
        combined_binary_rejections_dfs__compact.append(combined_binary_rejections_df__compact)
        combined_categorised_rejections_dfs__detailed.append(combined_categorised_rejections_df__detailed)
        combined_categorised_rejections_dfs__compact.append(combined_categorised_rejections_df__compact)
        combined_categorised2_rejections_dfs__detailed.append(combined_categorised2_rejections_df__detailed)
        combined_categorised2_rejections_dfs__compact.append(combined_categorised2_rejections_df__compact)

    # Concatenate all combined DataFrames for p-values and rejections
    final_pvalues_df__detailed = pd.concat(combined_pvalues_dfs__detailed, ignore_index=True)
    final_pvalues_df__compact = pd.concat(combined_pvalues_dfs__compact, ignore_index=True)
    final_binary_rejections_df__detailed = pd.concat(combined_binary_rejections_dfs__detailed, ignore_index=True)
    final_binary_rejections_df__compact = pd.concat(combined_binary_rejections_dfs__compact, ignore_index=True)
    final_categorised_rejections_df__detailed = pd.concat(combined_categorised_rejections_dfs__detailed, ignore_index=True)
    final_categorised_rejections_df__compact = pd.concat(combined_categorised_rejections_dfs__compact, ignore_index=True)
    final_categorised2_rejections_df__detailed = pd.concat(combined_categorised2_rejections_dfs__detailed, ignore_index=True)
    final_categorised2_rejections_df__compact = pd.concat(combined_categorised2_rejections_dfs__compact, ignore_index=True)

    # Save these concatenated DataFrames to CSV
    final_pvalues_path__detailed = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--combined_statistical_tests_adjusted_pvalues--detailed__({dataset_name}).csv")
    final_pvalues_df__detailed.to_csv(final_pvalues_path__detailed, index=False)
    final_pvalues_path__compact = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--combined_statistical_tests_adjusted_pvalues--compact__({dataset_name}).csv")
    final_pvalues_df__compact.to_csv(final_pvalues_path__compact, index=False)
    final_binary_rejections_path__detailed = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--combined_statistical_tests_binary_rejections--detailed__({dataset_name}).csv")
    final_binary_rejections_df__detailed.to_csv(final_binary_rejections_path__detailed, index=False)
    final_binary_rejections_path__compact = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--combined_statistical_tests_binary_rejections--compact__({dataset_name}).csv")
    final_binary_rejections_df__compact.to_csv(final_binary_rejections_path__compact, index=False)
    final_categorised_rejections_path__detailed = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--combined_statistical_tests_categorised_rejections--detailed__({dataset_name}).csv")
    final_categorised_rejections_df__detailed.to_csv(final_categorised_rejections_path__detailed, index=False)
    final_categorised_rejections_path__compact = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--combined_statistical_tests_categorised_rejections--compact__({dataset_name}).csv")
    final_categorised_rejections_df__compact.to_csv(final_categorised_rejections_path__compact, index=False)
    final_categorised2_rejections_path__detailed = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--combined_statistical_tests_categorised2_rejections--detailed__({dataset_name}).csv")
    final_categorised2_rejections_df__detailed.to_csv(final_categorised2_rejections_path__detailed, index=False)
    final_categorised2_rejections_path__compact = os.path.join(bias_stats_dir_path, f"{model['shortname']}__all_tests--combined_statistical_tests_categorised2_rejections--compact__({dataset_name}).csv")
    final_categorised2_rejections_df__compact.to_csv(final_categorised2_rejections_path__compact, index=False)


def combine_simulation_results(dfs, method='mean_std'):
    """
    Combine results from multiple DataFrames based on the specified method.
    For 'binary_rejections', expects binary True/False entries in the last five columns; 
    for all other methods, expects p-values in these columns.
    """
    if method not in ['mean_std', 'binary_rejections', 'categorised_rejections', 'categorised2_rejections', 'fisher', 'stouffer']:
        raise ValueError("Invalid method specified. Choose from 'mean_std', 'binary_rejections', 'categorised_rejections', 'categorised2_rejections', 'fisher', 'stouffer'.")
    
    # Determine the expected data type for the check based on the method
    expected_type = 'binary' if method == 'binary_rejections' else 'p-value'

    # Perform data type check on the last five (relevant) columns which are expected to contain the relevant data
    for df in dfs:
        verify_bias_data(df[df.columns[-5:]], expected_type)

    # Copy the structure of the first DataFrame -- all dfs should have consistent structure
    combined_df_detailed = dfs[0].copy()
    combined_df_compact = dfs[0].copy()

    # Loop through each DataFrame and gather the relevant columns (last five columns)
    columns_of_interest = dfs[0].columns[-5:] 
    
    for col in columns_of_interest:
        column_data = np.array([df[col].values for df in dfs])  # array of values for the same column from each DataFrame df in dfs

        if method == 'mean_std':
            # Calculate mean and standard deviation
            avgs = np.mean(column_data, axis=0)
            stddevs = np.std(column_data, axis=0, ddof=1)  # ddof=1 makes it sample standard deviation
            percent_stddevs = np.where(avgs == 0, 0, (stddevs / avgs) * 100)  # avoid division by zero
            combined_df_detailed[col] = [{"Mean": m, "SD": s, "%SD": p} for m, s, p in zip(avgs, stddevs, percent_stddevs)]
            combined_df_compact[col] = [f"{m:.4g} ± {s:.4g} (± {p:.2f}%)" for m, s, p in zip(avgs, stddevs, percent_stddevs)]
        
        elif method == 'binary_rejections':
            # Calculate majority vote (assuming binary values, i.e., booleans True or False)
            majority_vote = np.mean(column_data, axis=0) > 0.5
            total_count = len(dfs)
            true_counts = np.sum(column_data, axis=0)
            false_counts = total_count - true_counts
            max_digits_count = len(str(total_count))
            max_percentage_length = 6  # "xxx.xx" is 6 characters at most for 100.00%
            combined_df_detailed[col] = [f"False: {f:{max_digits_count}} [{f/total_count*100:{max_percentage_length}.2f}%] | True: {t:{max_digits_count}} [{t/total_count*100:{max_percentage_length}.2f}%]" for t, f in zip(true_counts, false_counts)]
            combined_df_compact[col] = majority_vote

        elif method == 'categorised_rejections':
            # Classify each p-value and count occurrences in each category
            counts = np.array([[sum((p >= 0.05) for p in pvals), sum((0.001 <= p < 0.05) for p in pvals), sum((p < 0.001) for p in pvals)] for pvals in column_data.T])  # pvals represent a list of all the p-values in the same entry location in the dfs
            categories = ['(p ≥ 0.05)', '(0.001 ≤ p < 0.05)', '(p < 0.001)']
            max_cat_length = max(len(cat) for cat in categories)  # determine the maximum length of category strings for alignment
            max_count_length = max(len(str(count)) for counts_row in counts for count in counts_row)  # Find the maximum length of count strings for alignment
            combined_df_detailed[col] = ["\n".join(f"{cat:{max_cat_length}} : {count:{max_count_length}} [{count/sum(counts_row)*100:.2f}%]" for cat, count in zip(categories, counts_row)) for counts_row in counts]
            combined_df_compact[col] = [categories[i] for i in np.argmax(counts, axis=1)]

        elif method == 'categorised2_rejections':
            # Classify each p-value and count occurrences in each category
            counts = np.array([[sum((p >= 0.05) for p in pvals), sum((0.001 <= p < 0.05) for p in pvals), sum((p < 0.001) for p in pvals)] for pvals in column_data.T])  # pvals represent a list of all the p-values in the same entry location in the dfs
            categories = ['FALSE', 'TRUE', 'TRUE+']
            max_cat_length = max(len(cat) for cat in categories)  # determine the maximum length of category strings for alignment
            max_count_length = max(len(str(count)) for counts_row in counts for count in counts_row)  # Find the maximum length of count strings for alignment
            combined_df_detailed[col] = ["\n".join(f"{cat:{max_cat_length}} : {count:{max_count_length}} [{count/sum(counts_row)*100:.2f}%]" for cat, count in zip(categories, counts_row)) for counts_row in counts]
            combined_df_compact[col] = [categories[i] for i in np.argmax(counts, axis=1)]
        
        elif method == 'fisher' or method == 'stouffer':  # Note that this has not been used in the current code and analysis
            # Combine the p-values using the specified method for each column
            _, p_values = combine_pvalues(column_data, method=method)
            combined_df_detailed[col] = p_values
            combined_df_compact[col] = [f"P-value: {p:.4g}" for p in p_values]

    return combined_df_detailed, combined_df_compact

# =======================================================
# ======= BIAS - STATS SIMULATION FUNCTIONS - END =======
# =======================================================




def parse_args():
    parser = argparse.ArgumentParser(description="Model bias inspection.")
    parser.add_argument('--model_dir', help="Path to the model's outputs directory")
    parser.add_argument('--model_shortname', help="Model's short name")
    parser.add_argument('--config', default='chexpert', choices=['chexpert', 'mimic'], help='Config dataset module to use')
    parser.add_argument('--labels', nargs='+', default=['Pleural Effusion', 'Others', 'No Finding'], 
                        help='List of labels to process')
    parser.add_argument('--local_execution', type=bool, default=True, help='Boolean to check whether the code is run locally or remotely')
    parser.add_argument('--kd_type_subdir_name', default='', 
                        help='Optional subdir name denoting the type of knowledge distillation applied to the model in question, to extend global directory paths with')
    parser.add_argument('--std_centred', action='store_true', 
                        help='When provided, is set to true to center plot axes around the standard deviation of the data')
    parser.add_argument('--stat_test_types', nargs='+', default=['ks'], 
                        choices=['ks', 'ks_permutation', 'mannwhitney', 'anderson', 'median', 'mood', 'kruskal', 'cramervonmises'],
                        help='List of statistical tests to perform')
    return parser.parse_args()



if __name__ == "__main__":

    args = parse_args()
    # Load the configuration dynamically based on the command line argument
    config = load_config(args.config)
    # Accessing the configuration to import dataset-specific variables
    dataset_name = get_dataset_name(args.config)

    if args.local_execution:
        # Construct local file path dynamically based on the dataset name
        local_base_path = '/Users/macuser/Desktop/Imperial/70078_MSc_AI_Individual_Project/code/Test_Records/Test_Resample'
        local_dataset_path = os.path.join(local_base_path, dataset_name)
        local_filename = os.path.basename(config.TEST_RECORDS_CSV)  # Extracts filename from the config path
        TEST_RECORDS_CSV = os.path.join(local_dataset_path, local_filename)
    else:
        TEST_RECORDS_CSV = config.TEST_RECORDS_CSV

    data_characteristics = pd.read_csv(TEST_RECORDS_CSV)

    # Path to base output directory
    base_output_path = os.getcwd()  # Use the current working directory for outputs
    global_bias_inspection_dir_path = os.path.join(base_output_path, f'bias_inspection2--{dataset_name}')

    # 'Global' directories to save bias inspection outputs
    global_combined_joint_plots_dir_path = os.path.join(global_bias_inspection_dir_path, 'models_combined_joint_plots/')
    global_combined_joint_plots_dir_path__main_samples = os.path.join(global_combined_joint_plots_dir_path, 'main_samples/', args.kd_type_subdir_name)
    global_combined_joint_plots_dir_path__all_unique_data_points = os.path.join(global_combined_joint_plots_dir_path, 'all_unique_data_points/', args.kd_type_subdir_name)
    global_combined_marginal_plots_dir_path = os.path.join(global_bias_inspection_dir_path, 'models_combined_marginal_plots/')
    global_combined_marginal_plots_dir_path__main_samples = os.path.join(global_combined_marginal_plots_dir_path, 'main_samples/', args.kd_type_subdir_name)
    global_combined_marginal_plots_dir_path__all_unique_data_points = os.path.join(global_combined_marginal_plots_dir_path, 'all_unique_data_points/', args.kd_type_subdir_name)
    # Ensure these directories exist
    os.makedirs(global_bias_inspection_dir_path, exist_ok=True)
    os.makedirs(global_combined_joint_plots_dir_path, exist_ok=True)
    os.makedirs(global_combined_joint_plots_dir_path__main_samples, exist_ok=True)
    os.makedirs(global_combined_joint_plots_dir_path__all_unique_data_points, exist_ok=True)
    os.makedirs(global_combined_marginal_plots_dir_path, exist_ok=True)
    os.makedirs(global_combined_marginal_plots_dir_path__main_samples, exist_ok=True)
    os.makedirs(global_combined_marginal_plots_dir_path__all_unique_data_points, exist_ok=True)



    # Dict to store model's info
    model_dir = args.model_dir
    model_fullname = os.path.basename(model_dir)
    model_shortname = args.model_shortname
    model = {
        "directory": model_dir,
        "fullname" : model_fullname,
        "shortname": model_shortname
        }

    # Path to outputs and data characteristics files
    embeddings_csv_filepath = os.path.join(model["directory"], 'embeddings_test.csv')
    model_embeddings = read_csv_file(embeddings_csv_filepath)

    print(f"data_characteristics shape (BEFORE clean up for duplicates): {data_characteristics.shape}")
    print(f"model_embeddings shape (BEFORE clean up for duplicates): {model_embeddings.shape}")

    # The rows in data_characteristics directly correspond to rows in model_embeddings; Resetting indices for both in case they are misaligned
    data_characteristics.reset_index(drop=True, inplace=True)
    model_embeddings.reset_index(drop=True, inplace=True)

    # Drop duplicates from data_characteristics and keep the indices of the remaining rows
    data_characteristics_clean = data_characteristics.drop_duplicates(keep='first').copy()
    indices_to_keep = data_characteristics_clean.index

    # Filter model_embeddings to only keep rows that match the indices from data_characteristics
    model_embeddings_clean = model_embeddings.iloc[indices_to_keep].copy()

    print(f"data_characteristics shape (AFTER clean up for duplicates): {data_characteristics_clean.shape}")
    print(f"model_embeddings shape (AFTER clean up for duplicates): {model_embeddings_clean.shape}")

    # Verify the shapes are still matching
    assert len(data_characteristics_clean) == len(model_embeddings_clean), "The dataframes should have the same length after filtering for duplicates."

    # Get embeddings size
    num_features = get_num_features(model_embeddings_clean)

    # Get embeddings columns
    embed_cols_names = [f'embed_{i}' for i in range(1, num_features+1)]
    embeds = model_embeddings_clean[embed_cols_names].to_numpy()
    n, m = embeds.shape
    print(f"Embeddings shape: {embeds.shape}")



    ## Directories to save bias analysis outputs, directly inside the model's main directory
    model_bias_directories = setup_model_bias_analysis_directories(model_directory=model["directory"],
                                                                   mainsample_subdirectory_name=MAIN_SAMPLE_SUBDIR_NAME,
                                                                   alluniquedata_subdirectory_name=ALL_UNIQUE_DATA_SUBDIR_NAME,
                                                                   simulation_subdirectory_name=SIMULATION_SUBDIR_NAME)
    # Main Bias Analysis Directory
    bias_dir_path = model_bias_directories['bias_dir_path']
    # Statistics Directories
    bias_stats_dir_path = model_bias_directories['bias_stats_dir_path']
    bias_stats_dir_path__main_sample = model_bias_directories['bias_stats_dir_path__main_sample']
    bias_stats_dir_path__all_unique_data_points = model_bias_directories['bias_stats_dir_path__all_unique_data_points']
    bias_stats_dir_path__simulation = model_bias_directories['bias_stats_dir_path__simulation']
    # PCA Analysis Directories
    pca_dir_path = model_bias_directories['pca_dir_path']
    pca_plots_joint_dir_path = model_bias_directories['pca_plots_joint_dir_path']
    pca_plots_joint_dir_path__main_sample = model_bias_directories['pca_plots_joint_dir_path__main_sample']
    pca_plots_joint_dir_path__all_unique_data_points = model_bias_directories['pca_plots_joint_dir_path__all_unique_data_points']
    pca_plots_marginal_dir_path = model_bias_directories['pca_plots_marginal_dir_path']
    pca_plots_marginal_dir_path__main_sample = model_bias_directories['pca_plots_marginal_dir_path__main_sample']
    pca_plots_marginal_dir_path__all_unique_data_points = model_bias_directories['pca_plots_marginal_dir_path__all_unique_data_points']
    # t-SNE Analysis Directories
    tsne_dir_path = model_bias_directories['tsne_dir_path']
    tsne_plots_joint_dir_path = model_bias_directories['tsne_plots_joint_dir_path']
    tsne_plots_joint_dir_path__main_sample = model_bias_directories['tsne_plots_joint_dir_path__main_sample']
    tsne_plots_joint_dir_path__all_unique_data_points = model_bias_directories['tsne_plots_joint_dir_path__all_unique_data_points']
    tsne_plots_marginal_dir_path = model_bias_directories['tsne_plots_marginal_dir_path']
    tsne_plots_marginal_dir_path__main_sample = model_bias_directories['tsne_plots_marginal_dir_path__main_sample']
    tsne_plots_marginal_dir_path__all_unique_data_points = model_bias_directories['tsne_plots_marginal_dir_path__all_unique_data_points']
    # Combined Plots Directories
    combined_plots_dir_path = model_bias_directories['combined_plots_dir_path']
    combined_joint_plots_dir_path = model_bias_directories['combined_joint_plots_dir_path']
    combined_joint_plots_dir_path__main_sample = model_bias_directories['combined_joint_plots_dir_path__main_sample']
    combined_joint_plots_dir_path__all_unique_data_points = model_bias_directories['combined_joint_plots_dir_path__all_unique_data_points']
    combined_marginal_plots_dir_path = model_bias_directories['combined_marginal_plots_dir_path']
    combined_marginal_plots_dir_path__main_sample = model_bias_directories['combined_marginal_plots_dir_path__main_sample']
    combined_marginal_plots_dir_path__all_unique_data_points = model_bias_directories['combined_marginal_plots_dir_path__all_unique_data_points']



    # Get PCA embeddings
    embeds_pca, exp_var = apply_pca(embeds=embeds, df=data_characteristics_clean, pca_dir_path=pca_dir_path)
    # Get TSNE embeddings from the PCA-reduced embeddings
    embeds_tsne = apply_tsne(embeds_pca=embeds_pca, df=data_characteristics_clean, tsne_dir_path=tsne_dir_path)


    ## Some pre-processing of the newly updated (from PCA and t-SNE above) data_characteristics_clean DataFrame, to also be saved
    # Categorise age into bins for better demographic analysis and visualisation
    data_characteristics_clean['binned_age'] = data_characteristics_clean['age'].apply(bin_age)

    # Replace 'Other' with 'Others' in the 'disease' column
    data_characteristics_clean['disease'] = data_characteristics_clean['disease'].replace('Other', 'Others')

    # Replicate entries for having capital letters in plots (duplicate columns)
    data_characteristics_clean['Disease'] = data_characteristics_clean['disease']
    data_characteristics_clean['Sex'] = data_characteristics_clean['sex']
    data_characteristics_clean['Age'] = data_characteristics_clean['binned_age']
    data_characteristics_clean['Race'] = data_characteristics_clean['race']

    # Save the updated DataFrame to a CSV file for potential further analysis
    csv_filename_all_data = f"{model['shortname']}__inspection_all_unique_data__({dataset_name}).csv"
    output_csv_path = os.path.join(bias_dir_path, csv_filename_all_data)
    data_characteristics_clean.to_csv(output_csv_path)
    print(f"Updated unique (cleaned) data characteristics saved to: {output_csv_path}")


    # Prepare a sample of data for bias analysis, drawing N_SAMPLES from each racial subgroup
    csv_filename_sample = f"{model['shortname']}__inspection_sample__({dataset_name}).csv"
    sample_df = stratified_sample_by_race(
        df=data_characteristics_clean, 
        n_samples=N_SAMPLES, 
        random_seed=RANDOM_SEED, 
        strata_columns=STRATA_COLUMNS,
        race_column="race",
        output_dir=bias_dir_path, 
        csv_filename=csv_filename_sample, 
        save_sample=True)
    sample_df = sample_df.sample(frac=1, random_state=RANDOM_SEED) # shuffle data for proper visualisation due to overlapping following z order of data points in plot
    
    # Create a comprehensive sample using all available unique data points for further analysis 
    # (note: prevalences across subgroups—race, disease, sex, age—are not preserved in this sample).
    all_unique_data_points_df = data_characteristics_clean.sample(frac=1, random_state=RANDOM_SEED)



    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # ==== BIAS ANALYSIS - GENERATE PLOTS & STATS ====
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

    # Analysis with sampled data
    print("Starting bias analysis with main sampled data...")
    perform_bias_inspection(
        sample_df=sample_df, 
        labels=args.labels, 
        pca_plots_joint_dir_path=pca_plots_joint_dir_path__main_sample, 
        pca_plots_marginal_dir_path=pca_plots_marginal_dir_path__main_sample, 
        tsne_plots_joint_dir_path=tsne_plots_joint_dir_path__main_sample, 
        tsne_plots_marginal_dir_path=tsne_plots_marginal_dir_path__main_sample, 
        std_centred=args.std_centred,
        bias_dir_path=bias_dir_path, 
        analysis_subdir_name=MAIN_SAMPLE_SUBDIR_NAME,
        model=model, 
        combined_joint_plots_dir_path=combined_joint_plots_dir_path__main_sample, 
        global_combined_joint_plots_dir_path=global_combined_joint_plots_dir_path__main_samples,
        combined_marginal_plots_dir_path=combined_marginal_plots_dir_path__main_sample, 
        global_combined_marginal_plots_dir_path=global_combined_marginal_plots_dir_path__main_samples, 
        dataset_name=dataset_name
        )
    perform_bias_statistical_analysis(
        sample_df=sample_df, 
        bias_stats_dir_path=bias_stats_dir_path__main_sample, 
        exp_var=exp_var, 
        model=model, 
        dataset_name=dataset_name,
        stat_test_types=args.stat_test_types,
        diseases=['Pleural Effusion', 'No Finding']
        )

    # Analysis with all unique data points
    print("Starting bias analysis with all data points...")
    perform_bias_inspection(
        sample_df=all_unique_data_points_df, 
        labels=args.labels, 
        pca_plots_joint_dir_path=pca_plots_joint_dir_path__all_unique_data_points, 
        pca_plots_marginal_dir_path=pca_plots_marginal_dir_path__all_unique_data_points, 
        tsne_plots_joint_dir_path=tsne_plots_joint_dir_path__all_unique_data_points, 
        tsne_plots_marginal_dir_path=tsne_plots_marginal_dir_path__all_unique_data_points, 
        std_centred=args.std_centred,
        bias_dir_path=bias_dir_path, 
        analysis_subdir_name=ALL_UNIQUE_DATA_SUBDIR_NAME,
        model=model, 
        combined_joint_plots_dir_path=combined_joint_plots_dir_path__all_unique_data_points, 
        global_combined_joint_plots_dir_path=global_combined_joint_plots_dir_path__all_unique_data_points,
        combined_marginal_plots_dir_path=combined_marginal_plots_dir_path__all_unique_data_points, 
        global_combined_marginal_plots_dir_path=global_combined_marginal_plots_dir_path__all_unique_data_points, 
        dataset_name=dataset_name
        )
    perform_bias_statistical_analysis(
        sample_df=all_unique_data_points_df,
        bias_stats_dir_path=bias_stats_dir_path__all_unique_data_points,
        exp_var=exp_var,
        model=model,
        dataset_name=dataset_name,
        stat_test_types=args.stat_test_types,
        diseases=['Pleural Effusion', 'No Finding']
        )
    
    # Analysis through simulation with multiple samples
    print("Starting bias analysis simulation with multiple samples...")
    # Generate N_SIMULATION_ITERATIONS samples with varying seeds
    samples_dfs = generate_multiple_samples(
        df=data_characteristics_clean, 
        n_samples=N_SAMPLES, 
        n_iterations=N_SIMULATION_ITERATIONS,
        strata_columns=STRATA_COLUMNS,
        race_column="race"
        )
    # Perform statistical analysis simulation: calculate mean/std of p-values and majority vote for rejections.
    simulate_bias_statistical_analysis(
        samples_dfs=samples_dfs, 
        bias_stats_dir_path=bias_stats_dir_path__simulation, 
        exp_var=exp_var, 
        model=model, 
        dataset_name=dataset_name,
        stat_test_types=args.stat_test_types,
        diseases=['Pleural Effusion', 'No Finding']  # Diseases to compare, currently set to compare 'Pleural Effusion' and 'No Finding' only
        )
    
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    # ==== BIAS ANALYSIS - GENERATE PLOTS & STATS ====
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<





