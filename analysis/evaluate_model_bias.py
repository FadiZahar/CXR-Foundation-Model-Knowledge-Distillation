import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.manifold import TSNE


# Import global shared variables
from config.config_shared import LABELS, RACES, SEXES
# Import the configuration loader
from config.loader_config import load_config, get_dataset_name

np.random.seed(42)



def parse_args():
    parser = argparse.ArgumentParser(description="Model bias inspection.")
    parser.add_argument('--outputs_dir', type=str, required=True, help='Path to outputs directory')
    parser.add_argument('--config', default='chexpert', choices=['chexpert', 'mimic'], help='Config dataset module to use')
    parser.add_argument('--labels', nargs='+', default=["No Finding", "Cardiomegaly", "Pneumothorax", "Pleural Effusion"], 
                        help='List of labels to process')
    parser.add_argument('--focus_metric', default="Youden\'s Index", help='Performance metric to focus on for comparative analysis')
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    # Load the configuration dynamically based on the command line argument
    config = load_config(args.config)
    # Accessing the configuration to import dataset-specific variables
    dataset_name = get_dataset_name(args.config)
    # TEST_RECORDS_CSV = config.TEST_RECORDS_CSV
    TEST_RECORDS_CSV = '/Users/macuser/Desktop/Imperial/70078_MSc_AI_Individual_Project/code/external/biomedia/biodata-data-chext_xray/meta/algorithmic_encoding/chexpert.sample.test.csv'

    # Path to outputs and data characteristics files
    outputs_csv_file = os.path.join(args.outputs_dir, 'outputs_test.csv')
    model_outputs = pd.read_csv(outputs_csv_file)
    data_characteristics = pd.read_csv(TEST_RECORDS_CSV)

