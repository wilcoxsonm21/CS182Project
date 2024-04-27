import os
import uuid
import yaml

from pathlib import Path
from typing import Tuple, Union

from box import Box

def merge_dictionaries(dict1, dict2):
    #print(type(dict1), type(dict2))
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(value, dict) and isinstance(dict1[key], dict):
                merge_dictionaries(dict1[key], value)
        else:
            dict1[key] = value

    return dict1


def get_edited_configs(original_config: Box, updates: dict) -> Box:
    """
    Takes original config and updates it with new values.
    """
    new_config = original_config.copy()
    return new_config.update(updates)

def conf_prepare_output_dir(conf: Box) -> Tuple[Box, Path]:
    """
    Prepare output directory for the model.
    """
    conf = conf.copy()
    run_id = conf.training.resume_id
    if run_id is None:
        run_id = str(uuid.uuid4())

    out_dir = os.path.join(conf.out_dir, run_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    conf.out_dir = out_dir

    with open(os.path.join(out_dir, "config.yaml"), "w") as yaml_file:
        yaml.dump(conf.to_dict(), yaml_file, default_flow_style=False)

    return conf, Path(out_dir)

def conf_prepare_preatrained_load_conf(conf: Box, pretrained_model_path: Path) -> Box:
    """
    Prepare pretrained model loading configuration.
    """
    conf.model.pretrained_model_dir = pretrained_model_path
    return conf

def _load_config(config_path: Path) -> dict:
    """
    Load config from the file.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Config file {config_path} not found.")

    if "inherit" in config:
        for path in config["inherit"]:
            if path is not None:
                config = merge_dictionaries(config, _load_config(config_path.parent / path))

    return config


def load_config(config_path: Union[Path, str]) -> Box:
    """
    Load config from the file.
    """
    config = _load_config(Path(config_path))
    return Box(config)



