# Configuration for project-wide settings

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

