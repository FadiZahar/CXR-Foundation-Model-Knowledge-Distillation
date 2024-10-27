# Configuration for project-wide settings
import matplotlib.pyplot as plt

# Model Parameters
IMAGE_SIZE = (224, 224)
CXRFM_EMBEDS_SIZE = 1376
NUM_CLASSES = 14
LABELS = [
    'No Finding',
    'Enlarged Cardiomediastinum',
    'Cardiomegaly',
    'Lung Opacity',
    'Lung Lesion',
    'Edema',
    'Consolidation',
    'Pneumonia',
    'Atelectasis',
    'Pneumothorax',
    'Pleural Effusion',
    'Pleural Other',
    'Fracture',
    'Support Devices'
    ]
LABELS_BY_RELEVANCE = [
    'Pleural Effusion',
    'No Finding',
    'Cardiomegaly',
    'Pneumothorax',
    'Atelectasis',
    'Consolidation',
    'Edema',
    'Pleural Other',
    'Enlarged Cardiomediastinum',
    'Pneumonia',
    'Lung Lesion',
    'Lung Opacity',
    'Fracture',
    'Support Devices'
    ]
RACES = ['White', 'Asian', 'Black']
SEXES = ['Male', 'Female']


# Training
EPOCHS = 20
NUM_WORKERS = 4
BATCH_SIZE = 128   # previously 150
LEARNING_RATE = 0.001


# Evaluation
TARGET_FPR = 0.2
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95


# Plotting
OUT_DPI = 500

# solid --> '-'
# dotted --> ':'
# dashed --> '--'
# dashdot --> '-.'

# Access the colormaps
tab20 = list(plt.get_cmap('tab20').colors)
tab20b = list(plt.get_cmap('tab20b').colors)
tab20c = list(plt.get_cmap('tab20c').colors)
accent = list(plt.get_cmap('Accent').colors)
dark2 = list(plt.get_cmap('Dark2').colors)


MODEL_STYLES = {
    'CXR-FM-LP': {'color': tab20[2], 'marker': 'o', 'linestyle': '-'},
    'CXR-Model-LP': {'color': tab20[6], 'marker': 'o', 'linestyle': '-.'},
    'CXR-Model-FFT': {'color': tab20[6], 'marker': 'o', 'linestyle': '-'},
    'DenseNet121-FFT': {'color': tab20b[14], 'marker': 'o', 'linestyle': '-'},
    'ResNet50-FFT': {'color': tab20b[15], 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (MSE)': {'color': tab20[10], 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (MSE)': {'color': tab20[10], 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (MSE)': {'color': tab20[10], 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (MSE)': {'color': tab20[10], 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (MAE)': {'color': tab20[4], 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (MAE)': {'color': tab20[4], 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (MAE)': {'color': tab20[4], 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (MAE)': {'color': tab20[4], 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (HuberLoss)': {'color': tab20b[0], 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (HuberLoss)': {'color': tab20b[0], 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (HuberLoss)': {'color': tab20b[0], 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (HuberLoss)': {'color': tab20b[0], 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (CS)': {'color': tab20[14], 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (CS)': {'color': tab20[14], 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (CS)': {'color': tab20[14], 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (CS)': {'color': tab20[14], 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (MSE-CS Naive)': {'color': 'cadetblue', 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (MSE-CS Naive)': {'color': 'cadetblue', 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (MSE-CS Naive)': {'color': 'cadetblue', 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (MSE-CS Naive)': {'color': 'cadetblue', 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (MSE-CS Learned)': {'color': tab20b[9], 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (MSE-CS Learned)': {'color': tab20b[9], 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (MSE-CS Learned)': {'color': tab20b[9], 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (MSE-CS Learned)': {'color': tab20b[9], 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (MSE-CS | 0.5-0.5)': {'color': tab20b[16], 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (MSE-CS | 0.5-0.5)': {'color': tab20b[16], 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (MSE-CS | 0.5-0.5)': {'color': tab20b[16], 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (MSE-CS | 0.5-0.5)': {'color': tab20b[16], 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (MSE-CS | 0.6-0.4)': {'color': tab20[18], 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (MSE-CS | 0.6-0.4)': {'color': tab20[18], 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (MSE-CS | 0.6-0.4)': {'color': tab20[18], 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (MSE-CS | 0.6-0.4)': {'color': tab20[18], 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (MSE-CS | 0.7-0.3)': {'color': tab20b[5], 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (MSE-CS | 0.7-0.3)': {'color': tab20b[5], 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (MSE-CS | 0.7-0.3)': {'color': tab20b[5], 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (MSE-CS | 0.7-0.3)': {'color': tab20b[5], 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (MSE-CS | 0.8-0.2)': {'color': accent[6], 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (MSE-CS | 0.8-0.2)': {'color': accent[6], 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (MSE-CS | 0.8-0.2)': {'color': accent[6], 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (MSE-CS | 0.8-0.2)': {'color': accent[6], 'marker': 'o', 'linestyle': '-'},
    'CXR-FMKD-LP (MSE-CS | 0.9-0.1)': {'color': tab20c[0], 'marker': 'o', 'linestyle': ':'},
    'CXR-FMKD-FFT (MSE-CS | 0.9-0.1)': {'color': tab20c[0], 'marker': 'o', 'linestyle': '--'},
    'CXR-FMKD-Direct-LP (MSE-CS | 0.9-0.1)': {'color': tab20c[0], 'marker': 'o', 'linestyle': '-.'},
    'CXR-FMKD-Direct-FFT (MSE-CS | 0.9-0.1)': {'color': tab20c[0], 'marker': 'o', 'linestyle': '-'}
}