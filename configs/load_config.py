import yaml
import os
from configs.default import get_config as get_default_config


def get_config(mode_string):
    config_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs",
        f"{mode_string}_config.yml",
    )
    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)
    default_config = get_default_config()
    def update_config(default_config, config_dict):
        for k, v in config_dict.items():
            if isinstance(v, dict):
                if k not in default_config:
                    default_config[k] = {}
                update_config(default_config[k], v)
            else:
                default_config[k] = v
    
    update_config(default_config, config_dict)
    return default_config
