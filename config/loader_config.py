# Handling different configurations

def load_config(config_name):
    """Load the configuration based on the input name."""
    if config_name == 'mimic':
        from config import config_mimic as config_module
    else:
        from config import config_chexpert as config_module
    
    return config_module

def get_dataset_name(config_name):
    """Returns the dataset name based on the configuration key."""
    if config_name == 'chexpert':
        return 'CheXpert'
    elif config_name == 'mimic':
        return 'MIMIC'
    else:
        raise ValueError(f"Unknown configuration dataset key: {config_name}")