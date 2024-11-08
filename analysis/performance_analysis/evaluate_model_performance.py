import os
import argparse
import numpy as np
from PIL import Image

from matplotlib import font_manager, gridspec
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Import global shared variables
from config.config_shared import MODEL_STYLES, OUT_DPI

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


def wrap_labels(labels, first_wrap_length=8, subsequent_wrap_length=20):
    """Wrap labels to a new line before exceeding specified lengths."""
    wrapped_labels = []
    for label in labels:
        label = 'Macro Average' if label == 'Macro' else label
        parts = label.split()  # Split label into words
        wrapped_label = parts[0]  # Start with the first word

        # First wrapping point
        for i, part in enumerate(parts[1:]):  
            if len(wrapped_label + ' ' + part) <= first_wrap_length:
                wrapped_label += ' ' + part  # Add part if under first limit
            else:
                remaining_parts = parts[i+1:]  # Remainder of the parts
                wrapped_label += '\n' + ' '.join(remaining_parts)  # Add newline before part if over first limit
                # Check if further wrapping is needed
                if len(' '.join(remaining_parts)) > subsequent_wrap_length:
                    wrapped_label = rewrap_long_line(wrapped_label, subsequent_wrap_length)
                break

        wrapped_labels.append(wrapped_label)
    return wrapped_labels


def rewrap_long_line(text, max_length):
    """Re-wrap a single line that exceeds the max_length after a newline."""
    lines = text.split('\n')
    new_lines = [lines[0]]  # First line is already within the first_wrap_length
    words = lines[1].split()  # Work with the second line

    current_line = ""
    for word in words:
        if len(current_line + ' ' + word) <= max_length:
            current_line += ' ' + word if current_line else word  # Add space if not the first word in line
        else:
            if current_line:  # Push the current line to the list and reset
                new_lines.append(current_line)
            current_line = word  # Start a new line with the current word
    if current_line:  # Add any remaining line
        new_lines.append(current_line)

    return '\n'.join(new_lines)


def save_legend_image(legend_handles, legend_labels, output_dir, num_columns=3, filename="legend.png"):
    fig, ax = plt.subplots()  # Dummy figure for legend
    ax.axis('off')
    
    # Save the legend
    legend = ax.legend(
        legend_handles, legend_labels, 
        loc='center', 
        ncol=num_columns, 
        prop={'weight': 'bold'}, 
        handlelength=4, 
        fancybox=True,
        edgecolor='gainsboro',
        )
    for text, color in zip(legend.get_texts(), [model['color'] for model in models]):  # Skip last item (note)
        text.set_color(color)

    # Calculate text position based on the legend's bounding box
    # fig.transFigure coordinates are from (0,0) in bottom left to (1,1) in top right
    note_y_position = legend.get_window_extent().ymax / fig.dpi / fig.get_figheight()
    
    # Adding an explanatory note inside the legend figure
    note_text='Note: Lines indicate averages; shaded areas indicate SD'
    fig.text(1.4, note_y_position, note_text, horizontalalignment='right', verticalalignment='bottom', 
             color='dimgrey', fontstyle='italic', fontsize=12, transform=fig.transFigure)

    fig.canvas.draw()
    
    # Save as image
    legend_image_path = os.path.join(output_dir, filename)
    plt.savefig(legend_image_path, dpi=OUT_DPI, bbox_inches='tight')
    plt.close()

    return legend_image_path


def crop_plots(plot_dir, metrics, models, dataset_name, result_type, keep_xticks_for, crop_margin=35):
    for metric in metrics:
        original_path = os.path.join(plot_dir, f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_plot--{result_type}__({dataset_name}).png")
        cropped_path = os.path.join(plot_dir, f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_plot_CROPPED--{result_type}__({dataset_name}).png")

        with Image.open(original_path) as img:
            width, height = img.size
            pixels = img.load()

            # Determine the vertical center of the image
            vertical_center = height // 2

            # Scan horizontally from the center to the right to find the first non-white pixel
            right_bound = width - 1
            for x in range(width)[::-1]:
                if pixels[x, vertical_center][:-1] != (255, 255, 255):  # Assuming the background is white
                    right_bound = x
                    break

            # Find the top and bottom non-white pixel bounds within the found vertical line
            top_bound = 0
            bottom_bound = height - 1

            for y in range(height):
                if pixels[right_bound, y][:-1] != (255, 255, 255):
                    top_bound = y
                    break

            for y in range(height - 1, -1, -1):
                if pixels[right_bound, y][:-1] != (255, 255, 255):
                    bottom_bound = y
                    break

            # Apply a margin
            top_bound = max(0, top_bound - crop_margin)
            bottom_bound = min(height, bottom_bound + crop_margin) if metric != keep_xticks_for else height

            # Define the crop region (left, top, right, lower)
            crop_rectangle = (0, top_bound, width, bottom_bound)
            cropped_img = img.crop(crop_rectangle)
            cropped_img.save(cropped_path, dpi=(OUT_DPI, OUT_DPI))  # Maintain the original DPI
            print(f"Cropped image saved to: {cropped_path}")


def add_legend_to_plot(output_dir, models, results_df_key, metric, figsize_width=20, font_size=12, title_font_delta=4, legend_image_path=None, 
                       dataset_name='CheXpert'):
    """Aggregates the figures into a grid and displays them."""
    result_type = results_df_key.split('_')[0]

    # Gather all the image heights to compute height ratios
    heights = []
    widths = []
    total_height = 0

    plot_filepath = os.path.join(output_dir, f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_plot--{result_type}__({dataset_name}).png")
    with Image.open(plot_filepath) as img:
        width, height = img.size
        heights.append(height)
        widths.append(width)
        total_height += height
    
    if legend_image_path:
        with Image.open(os.path.join(output_dir, legend_image_path)) as img:
            legend_width, legend_height = img.size
            heights.append(legend_height)
            widths.append(legend_width)
            total_height += legend_height

    # Calculate height ratios
    height_ratios = [h / total_height for h in heights]
    if legend_image_path:
        height_ratios[-1] = height_ratios[0]*0.75  # Adjust the height ratio for the legend 

    # Calculate the dynamic total figure height based on the width and height ratio.
    reference_width = widths[0]  # Use the first plot width as a reference for scaling.
    scaling_factor = figsize_width / reference_width
    dynamic_fig_height = total_height * scaling_factor

    # Set up the grid
    fig = plt.figure(figsize=(figsize_width, dynamic_fig_height))
    gs = gridspec.GridSpec(len(height_ratios), 1, height_ratios=height_ratios)  # Include space for title

    # Add the main metric plot in its designated space
    ax_subplot = fig.add_subplot(gs[0, 0])
    img = mpimg.imread(plot_filepath)
    ax_subplot.imshow(img)
    ax_subplot.axis('off')

    # Add the legend as an image in the last row if specified
    if legend_image_path:
        ax_legend = fig.add_subplot(gs[-1, 0])
        ax_legend.axis('off')
        img_legend = mpimg.imread(os.path.join(legend_image_path))
        ax_legend.imshow(img_legend)

    grid_plot_filename = f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_plot_w_legend--{result_type}__({dataset_name}).png"
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(output_dir, grid_plot_filename), dpi=OUT_DPI)
    plt.close()

    print(f"{metric} Plot with legend saved as: {grid_plot_filename}")


def aggregate_plots_into_grid(output_dir, models, results_df_key, figsize_width=20, font_size=12, title_font_delta=4, legend_image_path=None, 
                              dataset_name='CheXpert', metrics_ordered=["AUC-PR", "AUC-ROC"]):
    """Aggregates the figures into a grid and displays them."""
    result_type = results_df_key.split('_')[0]

    # Gather all the image heights to compute height ratios
    heights = []
    widths = []
    total_height = 0
    for metric in metrics_ordered:
        plot_filepath = os.path.join(output_dir, f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_plot_CROPPED--{result_type}__({dataset_name}).png")
        with Image.open(plot_filepath) as img:
            width, height = img.size
            heights.append(height)
            widths.append(width)
            total_height += height

    if legend_image_path:
        with Image.open(os.path.join(output_dir, legend_image_path)) as img:
            legend_width, legend_height = img.size
            heights.append(legend_height)
            widths.append(legend_width)
            total_height += legend_height

    # Calculate height ratios
    height_ratios = [h / total_height for h in heights]
    if legend_image_path:
        height_ratios[-1] = height_ratios[0]*1.25  # Adjust the height ratio for the legend 

    # Calculate the dynamic total figure height based on the width and height ratio.
    reference_width = widths[0]  # Use the first plot width as a reference for scaling.
    scaling_factor = figsize_width / reference_width
    dynamic_fig_height = total_height * scaling_factor

    # Set up the grid
    fig = plt.figure(figsize=(figsize_width, dynamic_fig_height))
    gs = gridspec.GridSpec(len(height_ratios)+1, 1, height_ratios=[0.01] + height_ratios)  # Include space for title

    # Title setup
    ax_title = fig.add_subplot(gs[0, 0], frameon=False)  # Span all columns in the first row
    title_suffix = "Across All Classes" if 'detailed_df' in results_df_key else "Across Most Significant Classes"
    ax_title.set_title(f"Performance Metrics | {dataset_name}\n{title_suffix}", fontsize=font_size+title_font_delta)
    ax_title.axis('off')

    # Plot each metric in the specified order
    for i, metric in enumerate(metrics_ordered):
        ax_subplot = fig.add_subplot(gs[i+1, 0])
        plot_filepath = os.path.join(output_dir, f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_plot_CROPPED--{result_type}__({dataset_name}).png")
        img = mpimg.imread(plot_filepath)
        ax_subplot.imshow(img)
        ax_subplot.axis('off')

    # Add the legend as an image in the last row if specified
    if legend_image_path:
        ax_legend = fig.add_subplot(gs[-1, 0])
        ax_legend.axis('off')
        img_legend = mpimg.imread(os.path.join(legend_image_path))
        ax_legend.imshow(img_legend)

    grid_plot_filename = f"{len(models)}-Models__All_Metrics--combined_performance_plots--{result_type}__({dataset_name}).png"
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.savefig(os.path.join(output_dir, grid_plot_filename), dpi=OUT_DPI)
    plt.close()

    print(f"Combined grid plot saved as: {grid_plot_filename}")


def plot_standard(models, save_dir, results_df_key='detailed_df', show_seeds=False, font_size=20, fig_size=(10,6), title_font_delta=4,
                  axis_font_delta=1.5, line_width=0.8, label_pad=12.5, title_pad=12.5, crop_margin=35, metrics_ordered=["AUC-PR", "AUC-ROC"]):
    result_type = results_df_key.split('_')[0]
    metrics = models[0][results_df_key]['Metric'].unique()
    classes = models[0][results_df_key]['Class'].unique()
    x_ticks = np.arange(len(classes))  # Generate x-ticks for classes
    dataset_name = models[0]['dataset_name']  # The dataset of the first model in the list is expected to be the primary dataset being analysed"

    # Adjust class labels to wrap text appropriately
    wrapped_classes = wrap_labels(classes)

    # Prepare to save legends separately
    legend_info = None

    for metric in metrics:
        plt.figure(figsize=fig_size)

        for model in models:
            # Filter data for the specific metric
            results_df = model[results_df_key]
            metric_data = results_df[results_df['Metric'] == metric]
            color = model['color']
            linestyle = model['linestyle']
            marker = model['marker']
            shortname = model['shortname']

            # Plot each individual multirun model seed (columns 3-7)
            if show_seeds:
                for i, seed_col in enumerate(metric_data.columns[2:7]):
                    label = f"{shortname} (Different Seeds)" if i == 0 else None  # Label only for the first seed to avoid repetition in legend
                    plt.plot(x_ticks, metric_data[seed_col], label=label, color=color, linestyle=linestyle, marker=marker, linewidth=line_width)

            # Plot average with standard deviation shading
            average = metric_data['Average']
            sd = metric_data['SD']
            average_label = None if show_seeds else f"{shortname}"
            plt.plot(x_ticks, average, label=average_label, color=color, linestyle=linestyle, linewidth=line_width*1.5, marker=marker)
            plt.fill_between(x_ticks, average - sd, average + sd, color=color, alpha=0.15)

        label_pad_adjusted = label_pad+16 if 'AUC' in metric else label_pad
        plt.ylabel(metric, fontsize=font_size+axis_font_delta, labelpad=label_pad_adjusted)
        plt.yticks(fontsize=font_size+axis_font_delta)
        # plt.xlabel("Class")
        plt.xticks(ticks=x_ticks, labels=wrapped_classes, rotation=45, ha="right", fontsize=font_size)

        # Add light vertical grid lines and customise grid and labels
        ax = plt.gca()
        for i, label in enumerate(ax.get_xticklabels()):
            if 'Macro' in label.get_text() or 'Average' in label.get_text():
                label.set_fontweight('bold')
                ax.axvline(x=i, color='black', linewidth=1.2, alpha=0.8, linestyle=(5, (10, 3)))  # Thicker line for specific labels
            else:
                ax.axvline(x=i, color='gray', linewidth=1, alpha=0.8, linestyle=(5, (10, 3)))  # Standard line for other labels

        # Formatting the plot with conditional title
        title_suffix = "Across All Classes" if results_df_key == 'detailed_df' else "Across Most Significant Classes"
        plt.title(f"{metric} | {dataset_name}\n{title_suffix}", fontsize=font_size+title_font_delta, pad=title_pad)

        # This is where we store handles and labels for the legend if not already done
        if not legend_info:
            legend_info = plt.gca().get_legend_handles_labels()
            h, l = legend_info
            legend_image_path = save_legend_image(legend_handles=h, legend_labels=l, output_dir=save_dir, filename=f"{len(models)}-Models__legend.png")

        plt.tight_layout()

        # Save the plot to the specified directory
        filename = os.path.join(save_dir, f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_plot--{result_type}__({dataset_name}).png")
        plt.savefig(filename, dpi=OUT_DPI)
        plt.close()

        print(f"{metric} Plot saved as: {filename}")

        add_legend_to_plot(output_dir=save_dir, models=models, results_df_key=results_df_key, metric=metric, figsize_width=fig_size[0], font_size=font_size, 
                           title_font_delta=title_font_delta, legend_image_path=legend_image_path, dataset_name=dataset_name)

    # Once all plots are saved, call the cropping function
    crop_plots(save_dir, metrics, models, dataset_name, result_type, "Youden's J Statistic at 20% FPR", crop_margin=crop_margin)

    aggregate_plots_into_grid(output_dir=save_dir, models=models, results_df_key=results_df_key, figsize_width=fig_size[0], font_size=font_size, 
                              title_font_delta=title_font_delta, legend_image_path=legend_image_path, dataset_name=dataset_name, 
                              metrics_ordered=metrics_ordered)
    
    return legend_image_path
    

def plot_parallel_coordinates(models, save_dir, results_df_key='detailed_df', font_size=20, fig_size=(10,6), show_seeds=False, title_font_delta=4, 
                              padding_percent=0.05, axis_font_delta=1.5, line_width=0.8, label_pad=30, title_pad=12.5, crop_margin=35, rounding=False, 
                              sd_alpha=0.1, legend_image_path=None, metrics_ordered=["AUC-PR", "AUC-ROC"]):
    result_type = results_df_key.split('_')[0]
    metrics = models[0][results_df_key]['Metric'].unique()
    classes = models[0][results_df_key]['Class'].unique()
    dataset_name = models[0]['dataset_name']  # The dataset of the first model in the list is expected to be the primary dataset being analysed"

    # Adjust class labels to wrap text appropriately
    wrapped_classes = wrap_labels(classes)

    for metric in metrics:
        fig, main_axis = plt.subplots(figsize=fig_size)

        seeds_data = []
        avg_data = []
        upperb_data = []
        lowerb_data = []
        plotting_data = []

        for model in models:
            # Filter data for the specific metric
            results_df = model[results_df_key]
            metric_data = results_df[results_df['Metric'] == metric]

            seeds = {f'seed{i+1}': metric_data[seed_col] for i, seed_col in enumerate(metric_data.columns[2:7])}
            avg = metric_data['Average']
            sd = metric_data['SD']
            upperb = avg + sd
            lowerb = avg - sd

            seeds_data.append(seeds)
            avg_data.append(avg)
            upperb_data.append(upperb)
            lowerb_data.append(lowerb)

            color = model['color']
            linestyle = model['linestyle']
            marker = model['marker']
            shortname = model['shortname']
            plotting_data.append({'color': color, 'linestyle': linestyle, 'marker': marker, 'shortname': shortname})

        min_vals = np.min(lowerb_data, axis=0)
        max_vals = np.max(upperb_data, axis=0)
        delta_vals = max_vals - min_vals
        padding = delta_vals * padding_percent
        ymins = min_vals - padding
        ymaxs = max_vals + padding
        if rounding:
            # Rounding to the nearest multiple of 0.05
            base=0.05
            ymins = base * np.floor(ymins / base)
            ymaxs = base * np.ceil(ymaxs / base)
        ydelta = ymaxs - ymins

        normalised_avg_data = (avg_data - ymins) / ydelta
        transformed_avg_data = normalised_avg_data * ydelta[0] + ymins[0]

        normalised_lowerb_data = (lowerb_data - ymins) / ydelta
        transformed_lowerb_data = normalised_lowerb_data * ydelta[0] + ymins[0]

        normalised_upperb_data = (upperb_data - ymins) / ydelta
        transformed_upperb_data = normalised_upperb_data * ydelta[0] + ymins[0]

        axes = [main_axis] + [main_axis.twinx() for i in range(len(avg_data[0]) - 1)]
        for i, ax in enumerate(axes):
            ax.set_ylim(ymins[i], ymaxs[i])
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            if ax != main_axis:
                ax.spines['left'].set_visible(False)
                ax.yaxis.set_ticks_position('right')
                ax.spines["right"].set_position(("axes", i / (len(avg_data[0]) - 1)))

            # Explicitly set y-ticks at evenly spaced intervals to cover the full range
            num_ticks = round(fig_size[1]/1.6)  # 1.6 is taken for 5 ticks for a fig of height 8
            tick_values = np.linspace(ymins[i], ymaxs[i], num_ticks)
            ax.set_yticks(tick_values)
            ax.set_yticklabels([f'{tick:.2f}' for tick in tick_values], fontsize=font_size*0.9)

        main_axis.set_xlim(0, len(avg_data[0]) - 1)
        main_axis.set_xticks(range(len(avg_data[0])))
        main_axis.set_xticklabels(wrapped_classes, rotation=45, ha="right", fontsize=font_size)
        main_axis.tick_params(axis='x', which='major', pad=20, length=0)  # remove x-axis tick marks with length=0
        main_axis.spines['right'].set_visible(False)

        # Apply bold formatting and thicken the corresponding axis for specific labels
        for i, label in enumerate(main_axis.get_xticklabels()):
            if 'Macro' in label.get_text() or 'Average' in label.get_text():
                label.set_fontweight('bold')
                axes[i].spines["right"].set_linewidth(1.5)  # Thicken the spine

        main_axis.set_ylabel(metric, fontsize=font_size+axis_font_delta, labelpad=label_pad)

        for j in range(len(models)):
            main_axis.plot(range(len(avg_data[0])), 
                           transformed_avg_data[j,:], 
                           color=plotting_data[j]['color'], 
                           linestyle=plotting_data[j]['linestyle'],
                           linewidth=line_width*1.5, 
                           marker=plotting_data[j]['marker'])
            main_axis.fill_between(range(len(avg_data[0])), 
                           transformed_lowerb_data[j],
                           transformed_upperb_data[j],
                           color=plotting_data[j]['color'],
                           alpha=sd_alpha)
            
        # Formatting the plot with conditional title
        title_suffix = "Across All Classes" if results_df_key == 'detailed_df' else "Across Most Significant Classes"
        main_axis.set_title(f"{metric} | {dataset_name}\n{title_suffix}", fontsize=font_size+title_font_delta, pad=title_pad*3)
        
        plt.tight_layout()

        # Save the plot to the specified directory
        filename = os.path.join(save_dir, f"{len(models)}-Models__{metric.replace(' ', '_')}--performance_plot--{result_type}__({dataset_name}).png")
        plt.savefig(filename, dpi=OUT_DPI)
        plt.close()

        print(f"{metric} Plot saved as: {filename}")

        add_legend_to_plot(output_dir=save_dir, models=models, results_df_key=results_df_key, metric=metric, figsize_width=fig_size[0], font_size=font_size, 
                           title_font_delta=title_font_delta, legend_image_path=legend_image_path, dataset_name=dataset_name)

    # Once all plots are saved, call the cropping function
    crop_plots(save_dir, metrics, models, dataset_name, result_type, "Youden's J Statistic at 20% FPR", crop_margin=crop_margin)

    aggregate_plots_into_grid(output_dir=save_dir, models=models, results_df_key=results_df_key, figsize_width=fig_size[0], font_size=font_size, 
                              title_font_delta=title_font_delta, legend_image_path=legend_image_path, dataset_name=dataset_name,
                              metrics_ordered=metrics_ordered)

    


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
    datasets = []  # To track unique datasets, as a list not a set to preserve order

    for seed_results_dirpath in args.models_seed_results_dirpath:
        model_shortname, dataset_name = get_model_info(seed_results_dirpath)
        detailed_df, focused_df = load_aggregated_results(directory=seed_results_dirpath, model_shortname=model_shortname, dataset_name=dataset_name)

        # Track unique datasets
        if dataset_name not in datasets:
            datasets.append(dataset_name)

        # Append the new model entry
        models.append({
            "seed_results_dirpath": seed_results_dirpath,
            "shortname": model_shortname,
            "dataset_name": dataset_name,
            "color": MODEL_STYLES[model_shortname]['color'],
            "marker": MODEL_STYLES[model_shortname]['marker'],  # Default marker
            "linestyle": MODEL_STYLES[model_shortname]['linestyle'],
            "detailed_df": detailed_df,
            "focused_df": focused_df
        })

    if len(datasets) > 2:
        raise ValueError(f"Only up to two distinct datasets are expected for model comparison, but {len(datasets)} datasets were found.")

    if len(datasets) == 2:
        marker_map = {datasets[0]: 'o', datasets[1]: '*'}
        shortname_extension_map = {datasets[0]: '', datasets[1]: '*'}

        # Update markers and shortnames based on dataset
        for model in models:
            model['marker'] = marker_map[model['dataset_name']]
            model['shortname'] += shortname_extension_map[model['dataset_name']]

    # Path to base output directory
    base_output_path = os.getcwd()  # Use the current working directory for outputs

    # Directories to save performance
    detailed_performance_dir_path = os.path.join(base_output_path, 'detailed_performance/')
    models_detailed_performance_dir_path = os.path.join(detailed_performance_dir_path, f'{len(models)}_models/')
    detailed_standardplot_dir_path = os.path.join(models_detailed_performance_dir_path, 'standard/')
    detailed_parallelcoordplot_dir_path = os.path.join(models_detailed_performance_dir_path, 'parallel_coordinates/')
    focused_performance_dir_path = os.path.join(base_output_path, 'focused_performance/')
    models_focused_performance_dir_path = os.path.join(focused_performance_dir_path, f'{len(models)}_models/')
    focused_standardplot_dir_path = os.path.join(models_focused_performance_dir_path, 'standard/')
    focused_parallelcoordplot_dir_path = os.path.join(models_focused_performance_dir_path, 'parallel_coordinates/')
    os.makedirs(detailed_performance_dir_path, exist_ok=True)
    os.makedirs(models_detailed_performance_dir_path, exist_ok=True)
    os.makedirs(detailed_standardplot_dir_path, exist_ok=True)
    os.makedirs(detailed_parallelcoordplot_dir_path, exist_ok=True)
    os.makedirs(focused_performance_dir_path, exist_ok=True)
    os.makedirs(models_focused_performance_dir_path, exist_ok=True)
    os.makedirs(focused_standardplot_dir_path, exist_ok=True)
    os.makedirs(focused_parallelcoordplot_dir_path, exist_ok=True)


    font_size = 19
    fig_size_detailed_standard = (20,8)
    fig_size_focused_standard = (20,7.5)
    # fig_size_detailed_standard = (20,25)
    # fig_size_focused_standard = (20,25)
    fig_size_detailed_parallelcoord = (20,8)
    fig_size_focused_parallelcoord = (20,7.5)
    title_font_delta = 5
    axis_font_delta = 2
    metrics_ordered = ["AUC-PR", "AUC-ROC", "Maximum Youden's J Statistic", "Youden's J Statistic at 20% FPR"]

    legend_image_path = plot_standard(
        models=models, 
        save_dir=detailed_standardplot_dir_path, 
        results_df_key='detailed_df', 
        fig_size=fig_size_detailed_standard,
        show_seeds=False, 
        font_size=font_size, 
        title_font_delta=title_font_delta, 
        axis_font_delta=axis_font_delta, 
        line_width=0.8, 
        crop_margin=35,
        metrics_ordered=metrics_ordered
        )
    plot_parallel_coordinates(
        models=models, 
        save_dir=detailed_parallelcoordplot_dir_path, 
        results_df_key='detailed_df',
        font_size=font_size,  
        fig_size=fig_size_detailed_parallelcoord,
        show_seeds=False, 
        title_font_delta=title_font_delta, 
        axis_font_delta=axis_font_delta, 
        line_width=0.8, 
        crop_margin=135,
        sd_alpha=0.1,
        legend_image_path=legend_image_path,
        metrics_ordered=metrics_ordered
        )
    
    legend_image_path = plot_standard(
        models=models, 
        save_dir=focused_standardplot_dir_path, 
        results_df_key='focused_df', 
        fig_size=fig_size_focused_standard,
        show_seeds=False, 
        font_size=font_size, 
        title_font_delta=title_font_delta, 
        axis_font_delta=axis_font_delta, 
        line_width=0.8, 
        crop_margin=35,
        metrics_ordered=metrics_ordered
        )
    plot_parallel_coordinates(
        models=models, 
        save_dir=focused_parallelcoordplot_dir_path, 
        results_df_key='focused_df', 
        font_size=font_size, 
        fig_size=fig_size_focused_parallelcoord,
        show_seeds=False, 
        title_font_delta=title_font_delta, 
        axis_font_delta=axis_font_delta, 
        line_width=0.8, 
        crop_margin=135,
        sd_alpha=0.1,
        legend_image_path=legend_image_path,
        metrics_ordered=metrics_ordered
        )


