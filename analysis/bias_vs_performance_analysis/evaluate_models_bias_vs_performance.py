import os
import argparse
import numpy as np
from PIL import Image

from matplotlib import font_manager, gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Import global shared variables
from config.config_shared import MODEL_STYLES, OUT_DPI
# Import the configuration loader
from config.loader_config import get_dataset_name

# Import other utils functions
from analysis.bias_analysis.evaluate_models_disease_prediction import read_csv_file

# Check if Latin Modern Roman (~LaTeX) is available, and set it; otherwise, use the default font
if 'Latin Modern Roman' in [f.name for f in font_manager.fontManager.ttflist]:
    plt.rcParams['font.family'] = 'Latin Modern Roman'

np.random.seed(42)



def get_model_info(seed_results_dirpath):
    # e.g., of the form: "aggregated_seed_results--CXR-FM-LP--CheXpert"
    dirname = os.path.basename(seed_results_dirpath)
    parts = dirname.split("--")
    model_shortname = parts[1]
    dataset_name = parts[-1]
    return model_shortname, dataset_name


def ensure_conform_fullname(fullname):
    seedsplit = fullname.split('seed')
    string_end = seedsplit[-1].split('_')[0]
    fullname = seedsplit[0] + 'seed' + string_end
    return fullname


def load_bias_scores_df(directory, model_shortname, dataset_name):
    """Load bias scores df from specified directory."""
    compact_bias_scores_path = os.path.join(directory, f'{model_shortname}__all_tests--compiled_bias_scores--compact__({dataset_name}).csv')
    if os.path.exists(compact_bias_scores_path):
        bias_scores_df = read_csv_file(compact_bias_scores_path)
    else:
        bias_scores_df = None
    return bias_scores_df


def load_performance_results_df(directory, model_shortname, dataset_name):
    """Load aggregated results metrics from specified directory."""
    performance_results_path = os.path.join(directory, f'aggregated_results_metrics_focused--{model_shortname}--{dataset_name}.csv')
    if os.path.exists(performance_results_path):
        performance_results_df = read_csv_file(performance_results_path)
    else:
        performance_results_df = None
    return performance_results_df


def save_legend_image(legend_handles, legend_labels, output_dir, num_columns=3, filename="legend.png"):
    fig, ax = plt.subplots(figsize=(1, 1))  # Dummy figure for legend
    ax.axis('off')

    # Modify each original handle to customise the legend 
    new_handles = []
    for handle in legend_handles:
        if isinstance(handle, Line2D):  # For line plots (i.e., line of best fit), keep the same properties
            new_handle = Line2D([], [], color=handle.get_color(), marker=handle.get_marker(),
                                linestyle=handle.get_linestyle(), markersize=handle.get_markersize(),
                                linewidth=1)  # Adjust linewidth back to a thinner width
            new_handles.append(new_handle)
        else:  # For other types of handles (i.e., scatter plots), recreate handle with desired markersize
            new_handle = Line2D([], [], color=handle.get_facecolor()[0], marker=handle.get_paths()[0],
                                linestyle='None', markersize=None, linewidth=0)  # Adjust markersize back to its default (smaller) size
            new_handles.append(new_handle)
        

    # Prepend numbers to legend labels
    numbered_legend_labels = [f"{i+1}. {label}" for i, label in enumerate(legend_labels)]
    
    # Save the legend
    legend = ax.legend(
        new_handles, numbered_legend_labels, 
        loc='center', 
        ncol=num_columns, 
        prop={'weight': 'bold'}, 
        fancybox=True,
        edgecolor='gainsboro',
        )
    for text, color in zip(legend.get_texts(), [model['color'] for model in models]): 
        text.set_color(color)

    fig.canvas.draw()
    
    # Save as image
    legend_image_path = os.path.join(output_dir, filename)
    plt.savefig(legend_image_path, dpi=OUT_DPI, bbox_inches='tight')
    plt.close()

    return legend_image_path


def aggregate_plots_into_grid(output_dir, models, figsize_width=20, font_size=12, title_font_delta=4, legend_image_path=None, 
                              dataset_name='CheXpert', metrics_ordered=["AUC-PR", "AUC-ROC"], legend_width_ratio=0.9, plot_best_fit=False):
    """Aggregates the figures into a grid and displays them."""
    # Gather all the images' info for ratio computations
    heights = []
    widths = []
    filename_suffix = '--w_best_fit' if plot_best_fit else ''

    for metric in metrics_ordered:
        # Path to the desired metric plot
        plot_filepath = os.path.join(output_dir, f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_vs_bias_plot{filename_suffix}__({dataset_name}).png")
        with Image.open(plot_filepath) as img:
            plot_width, plot_height = img.size
            widths.append(plot_width)
            heights.append(plot_height)

    if legend_image_path:
        with Image.open(legend_image_path) as img:
            legend_width, legend_height = img.size
            # Adjust legend dimensions to maintain uniform width with the main plots
            legend_aspect_ratio = legend_height / legend_width
            desired_width = 2 * widths[0] * legend_width_ratio  # e.g., 90% of a full row made out of two main plots of the same width
            new_legend_height = legend_aspect_ratio * desired_width
            heights.append(new_legend_height)

    # Calculate total height and height ratios
    column_heights = heights[::2]
    total_height = sum(column_heights)
    height_ratios = [h / total_height for h in column_heights]

    # Adjusting height_ratios to increase space between the first two rows
    if len(height_ratios) > 1:
        height_ratios.insert(1, 0.02)

    # Calculate the dynamic total figure height based on the width and height ratio.
    scaling_factor = figsize_width / (2 * widths[0])  # Scaling factor assumes two plots per row; thus total width of two plots
    dynamic_fig_height = total_height * scaling_factor

    # Set up the grid
    fig = plt.figure(figsize=(figsize_width, dynamic_fig_height+0.6))
    # Calculate grid dimensions
    grid_rows = len(height_ratios)
    grid_cols = 2
    gs = gridspec.GridSpec(grid_rows + 1, grid_cols, height_ratios=[0.02] + height_ratios)  # Include space for title

    # Title setup
    ax_title = fig.add_subplot(gs[0, :], frameon=False)  # Span all columns in the first row
    ax_title.set_title(f"Performance vs. Bias Plots | {dataset_name}", fontsize=font_size+title_font_delta)
    ax_title.axis('off')

    # Plot each metric in the specified order
    for i, metric in enumerate(metrics_ordered):
        ax_subplot = fig.add_subplot(gs[i//grid_cols*2 + 1, i%grid_cols])  # adjusting indices to account for title space and extra space in-between the first two rows
        plot_filepath = os.path.join(output_dir, f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_vs_bias_plot{filename_suffix}__({dataset_name}).png")
        img = mpimg.imread(plot_filepath)
        ax_subplot.imshow(img)
        ax_subplot.axis('off')

    # Add the legend as an image in the last row if specified
    if legend_image_path:
        ax_legend = fig.add_subplot(gs[-1, :])  # Last row, spanning all columns
        ax_legend.axis('off')
        img_legend = mpimg.imread(legend_image_path)
        ax_legend.imshow(img_legend)

    grid_plot_filename = f"{len(models)}-Models__All_Metrics--combined_performance_vs_bias_plots{filename_suffix}__({dataset_name}).png"
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(output_dir, grid_plot_filename), dpi=OUT_DPI)
    plt.close()

    print(f"Combined grid plot saved as: {grid_plot_filename}")


def plot_performance_vs_bias(models, save_dir, font_size=20, fig_size=(10,6), title_font_delta=4, axis_font_delta=1.5, label_pad=12.5, 
                             title_pad=12.5, metrics_ordered=["AUC-PR", "AUC-ROC"], legend_num_columns=3, legend_width_ratio=0.9, 
                             marker_size=100, plot_best_fit=False):
    metrics = models[0]["performance_results_df"]['Metric'].unique()
    class_of_interest = "Average Classes 1 to 7"
    dataset_name = models[0]['dataset_name']  # The dataset of the first model in the list is expected to be the primary dataset being analysed"

    # Prepare to save legends separately
    legend_info = None

    for metric in metrics:
        plt.figure(figsize=fig_size)
        x_all = []
        y_all = []

        for model in models:
            # Extract dataframes and parameters from model dictionary
            bias_scores_df = model["bias_scores_df"]
            performance_results_df = model["performance_results_df"]

            # Extract x and y values
            x = bias_scores_df.loc[bias_scores_df['Statistical Test Applied'] == 'ks', "Attributes Average (Combined Bias Score)"].iloc[0]
            y = performance_results_df.loc[(performance_results_df['Metric'] == metric) & (performance_results_df['Class'] == class_of_interest), 
                                           model["fullname"]].iloc[0]

            x_all.append(x)
            y_all.append(y)

            color = model['color']
            marker = model['marker_for_performancevsbias']
            markersize = marker_size
            shortname = model['shortname']

            # Plotting the points
            plt.scatter(x, y, color=color, marker=marker, s=markersize, label=shortname)

        # Conditionally calculate and plot the line of best fit
        if plot_best_fit and x_all and y_all: 
            z = np.polyfit(x_all, y_all, 1)  # (degree 1 polynomial)
            p = np.poly1d(z)
            # Generating a sorted array of x values for a smoother line
            x_sorted = np.linspace(min(x_all), max(x_all), num=len(x_all))
            y_sorted = p(x_sorted)
            # Plotting the line
            plt.plot(x_sorted, y_sorted, color='k', linestyle="--", label='Line of Best Fit')
            

        plt.xlabel("Combined Bias Score\n(Average Across the Race and Sex Attributes)", fontsize=font_size+axis_font_delta)
        plt.ylabel(f"{metric}\n(Average for Classes 1 to 7)", fontsize=font_size+axis_font_delta, labelpad=label_pad)
        plt.xticks(fontsize=font_size)  
        plt.yticks(fontsize=font_size) 
        plt.title(f"{metric} vs. Bias Score", fontsize=font_size+title_font_delta, pad=title_pad)

        # This is where we store handles and labels for the legend if not already done
        if not legend_info:
            legend_info = plt.gca().get_legend_handles_labels()
            h, l = legend_info
            legend_image_path = save_legend_image(legend_handles=h, legend_labels=l, output_dir=save_dir, num_columns=legend_num_columns, 
                                                  filename=f"{len(models)}-Models__legend.png")

        plt.tight_layout()

        # Save the plot to the specified directory
        filename_suffix = '--w_best_fit' if plot_best_fit else ''
        filename = f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_vs_bias_plot{filename_suffix}__({dataset_name}).png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=OUT_DPI)
        plt.close()

        print(f"({metric} vs. Bias) Plot saved as: {filepath}")

    aggregate_plots_into_grid(output_dir=save_dir, models=models, figsize_width=fig_size[0]*2, font_size=font_size, 
                              title_font_delta=title_font_delta, legend_image_path=legend_image_path, dataset_name=dataset_name, 
                              metrics_ordered=metrics_ordered, legend_width_ratio=legend_width_ratio, plot_best_fit=plot_best_fit)
    
    return legend_image_path

    

def parse_args():
    parser = argparse.ArgumentParser(description="Plot performance metrics from aggregated results.")
    parser.add_argument('--models_bias_scores_dirpath', nargs='+', required=True,
                        help="Paths to directories with combined bias scores results.")
    parser.add_argument('--models_seed_results_dirpath', nargs='+', required=True,
                        help="Paths to directories with aggregated seed results. Ordering should match with models_bias_scores_dirpath")
    return parser.parse_args()



if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # List to store models info and data
    models = []
    datasets = []  # To track unique datasets, as a list not a set to preserve order

    for bias_scores_dirpath, seed_results_dirpath in zip(args.models_bias_scores_dirpath, args.models_seed_results_dirpath, ):
        model_shortname, dataset_name = get_model_info(seed_results_dirpath)
        bias_scores_df = load_bias_scores_df(directory=bias_scores_dirpath, model_shortname=model_shortname, dataset_name=dataset_name)
        performance_results_df = load_performance_results_df(directory=seed_results_dirpath, model_shortname=model_shortname, dataset_name=dataset_name)

        model_fullname = bias_scores_df.loc[bias_scores_df['Statistical Test Applied'] == 'ks', 'Model Fullname'].iloc[0]
        model_fullname = ensure_conform_fullname(model_fullname)  # Make sure the model's full name has the expected format

        # Track unique datasets
        if dataset_name not in datasets:
            datasets.append(dataset_name)

        # Append the new model entry
        models.append({
            "bias_scores_dirpath": bias_scores_dirpath,
            "seed_results_dirpath": seed_results_dirpath,
            "fullname": model_fullname,
            "shortname": model_shortname,
            "dataset_name": dataset_name,
            "color": MODEL_STYLES[model_shortname]['color'],
            "marker": MODEL_STYLES[model_shortname]['marker'],  # Default marker
            "marker_for_performancevsbias": MODEL_STYLES[model_shortname]['marker_for_performancevsbias'],  # PerformancevsBias marker
            "markersize": MODEL_STYLES[model_shortname]['markersize'],  # Default marker
            "linestyle": MODEL_STYLES[model_shortname]['linestyle'],
            "bias_scores_df": bias_scores_df,
            "performance_results_df": performance_results_df
        })

    if len(datasets) > 1:
        raise ValueError(f"Only one dataset is expected for model comparison, but {len(datasets)} datasets were found.")

    # Path to base output directory
    base_output_path = os.getcwd()  # Use the current working directory for outputs

    # Directories to save performance
    normal_plots_dir_path = os.path.join(base_output_path, 'normal_plots/')
    models_normal_plots_dir_path = os.path.join(normal_plots_dir_path, f'{len(models)}_models/')
    fit_plots_dir_path = os.path.join(base_output_path, 'fit_plots/')
    models_fit_plots_dir_path = os.path.join(fit_plots_dir_path, f'{len(models)}_models/')
    os.makedirs(normal_plots_dir_path, exist_ok=True)
    os.makedirs(models_normal_plots_dir_path, exist_ok=True)
    os.makedirs(fit_plots_dir_path, exist_ok=True)
    os.makedirs(models_fit_plots_dir_path, exist_ok=True)


    font_size_plotforgrid = 20
    fig_size_plotforgrid = (10,6)
    title_font_delta = 6
    axis_font_delta = 1.5
    label_pad = 12.5
    title_pad = 12.5
    legend_num_columns = 3
    legend_width_ratio = 0.85
    marker_size = 100
    metrics_ordered = [
        "AUC-PR", 
        "AUC-ROC", 
        "Maximum Youden's J Statistic", 
        "Youden's J Statistic at 20% FPR"
        ]


    plot_performance_vs_bias(
        models=models, 
        save_dir=models_normal_plots_dir_path, 
        font_size=font_size_plotforgrid, 
        fig_size=fig_size_plotforgrid,
        title_font_delta=title_font_delta, 
        axis_font_delta=axis_font_delta, 
        label_pad=label_pad,
        title_pad=title_pad,
        metrics_ordered=metrics_ordered,
        legend_num_columns=legend_num_columns,
        legend_width_ratio=legend_width_ratio,
        marker_size=marker_size,
        plot_best_fit=False
        )
    
    plot_performance_vs_bias(
        models=models, 
        save_dir=models_fit_plots_dir_path, 
        font_size=font_size_plotforgrid,
        fig_size=fig_size_plotforgrid, 
        title_font_delta=title_font_delta, 
        axis_font_delta=axis_font_delta, 
        label_pad=label_pad,
        title_pad=title_pad,
        metrics_ordered=metrics_ordered,
        legend_num_columns=legend_num_columns,
        legend_width_ratio=legend_width_ratio,
        marker_size=marker_size,
        plot_best_fit=True
        )
    
    