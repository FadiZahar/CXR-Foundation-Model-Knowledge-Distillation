import os
import numpy as np
import matplotlib.pyplot as plt



def examine_dat_files(source_dir, num_files=10, output_dir=None):
    files = sorted([f for f in os.listdir(source_dir) if f.endswith('.dat')][:num_files])
    plt.figure(figsize=(12, 30))

    global_min = float('inf')
    global_max = float('-inf')
    embeddings = []

    # Gather all embeddings to find global min and max
    for file in files:
        file_path = os.path.join(source_dir, file)
        data = np.fromfile(file_path, dtype=np.float32)
        embeddings.append(data)
        global_min = min(global_min, np.min(data))
        global_max = max(global_max, np.max(data))

    range_val = global_max - global_min
    buffer = 0.1 * range_val if range_val > 0 else 0.1
    cmap = plt.get_cmap('viridis')

    for i, (file, embedding_array) in enumerate(zip(files, embeddings), 1):
        ax = plt.subplot(num_files, 1, i)
        color = cmap((np.mean(embedding_array) - global_min) / range_val)  # Use mean to determine colour
        plt.plot(embedding_array, color=color, label=f'{file} - Min: {np.min(embedding_array):.2f}, Max: {np.max(embedding_array):.2f}')
        plt.title(f"Embedding Plot of {file}")
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True)
        ax.set_ylim(global_min - buffer, global_max + buffer)
        plt.legend()

        print(f"File: {file}, Embedding Length: {len(embedding_array)}")
        print(f"Min Value: {np.min(embedding_array):.2f}, Max Value: {np.max(embedding_array):.2f}")
        print(f"Mean Value: {np.mean(embedding_array):.2f}, Std Dev: {np.std(embedding_array):.2f}")

    plt.tight_layout()

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'individual_embeddings_plots.png'))
        plt.close()
    else:
        plt.show()

    
def plot_combined_dat_files(source_dir, num_files=10):
    files = sorted([f for f in os.listdir(source_dir) if f.endswith('.dat')])[:num_files]
    plt.figure(figsize=(20, 12)) 

    global_min = float('inf')
    global_max = float('-inf')
    embeddings = []

    for file in files:
        file_path = os.path.join(source_dir, file)
        data = np.fromfile(file_path, dtype=np.float32)
        embeddings.append(data)
        global_min = min(global_min, np.min(data))
        global_max = max(global_max, np.max(data))

    range_val = global_max - global_min
    buffer = 0.1 * range_val if range_val > 0 else 0.1
    cmap = plt.get_cmap('viridis')

    ax = plt.gca()
    ax.set_prop_cycle('color', [cmap(i) for i in np.linspace(0, 1, num_files)])

    for i, (file, data) in enumerate(zip(files, embeddings), 1):
        plt.plot(data, label=f'{file} - Min: {np.min(data):.2f}, Max: {np.max(data):.2f}')

    plt.title("Combined Embedding Plots")
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    ax.set_ylim(global_min - buffer, global_max + buffer)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.tight_layout()

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'combined_embeddings_plots.png'))
        plt.close()
    else:
        plt.show()



if __name__ == '__main__':
    source_dir = '/vol/biomedic3/bglocker/mscproj24/fz221/data/cxrfm_embeddings/chexpert/cxr_numpy'
    output_dir = '/vol/biomedic3/bglocker/mscproj24/fz221/cxr-fmkd/utils/converter_utils'
    examine_dat_files(source_dir=source_dir, output_dir=output_dir)
    plot_combined_dat_files(source_dir=source_dir, output_dir=output_dir)




