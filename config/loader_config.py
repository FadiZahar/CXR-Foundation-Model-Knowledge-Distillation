# Handling different configurations

def load_config(config_name):
    """Load the configuration based on the input name."""
    if config_name == 'mimic':
        from config import config_mimic as config_module
    else:
        from config import config_chexpert as config_module
    
    return config_module

