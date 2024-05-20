import torch
from munch import Munch
from pathlib import Path
import yaml
from typing import NamedTuple, List

import models
from eval import eval_model, build_evals, get_relevant_baselines_for_degree

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


class LoadInfo(NamedTuple):
    path: Path
    step: int
    alternative_train_conf_path: Path = None
    alternative_pretrained_path: Path = None
    alternative_name: str = ""
    name_addon: str = ""

def get_config(config_path: Path):

    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        return Munch.fromDict(yaml.safe_load(fp))
    
def all_values_equal(actual_dict, key_dict):

    for key, value in key_dict.items():

        try:

            if isinstance(value, dict):
                if not all_values_equal(actual_dict[key], value):
                    return False
            else:
                if actual_dict[key] != value:
                    return False
                
        except KeyError:
            return False

    return True
    
def filter_runs(parent_dir: Path, filter_dict: dict) -> List[Path]:

    # Find child directories
    run_dirs = [d for d in parent_dir.iterdir() if d.is_dir()]

    # Filter runs
    relevant_dirs = []
    for d in run_dirs:
        conf_path = d / "config.yaml"
        conf = get_config(conf_path)
        
        if all_values_equal(conf, filter_dict):
            relevant_dirs.append(d)

    return relevant_dirs
                

def new_get_model_from_run(config, run_path: Path, step, device="cuda") -> torch.nn.Module:

    #config_path = run_path / "config.yaml"
    #print(config_path)
    #conf = get_config(config_path)

    model = models.build_model(config.model, device=device)

    if step == -1:
        state_path = run_path / "state.pt"
        state = torch.load(state_path, map_location=torch.device(device))
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = run_path / f"model_{step}.pt"
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)

    return model, config


def new_get_run_metrics(config, run_path: Path, step: int, device: str = "cuda", include_noise=True, ground_truth_loss=False, smoothing=0):

    # Get model
    model, _ = new_get_model_from_run(config, run_path, step, device=device)
    model = model.eval()

    # Set configuration    
    evaluation_kwargs = build_evals(config)
    standard_args = evaluation_kwargs["standard"]
    standard_args["task_sampler_kwargs"] = config.training.task_kwargs

    # Loop wanted degrees
    metrics = eval_model(model, include_noise=include_noise, ground_truth_loss=ground_truth_loss, smoothing=smoothing, device=device, **standard_args)

    return metrics

def baseline_data(train_conf_path: Path, include_noise=True, ground_truth_loss=False, smoothing=0, device="cuda"):

    config = get_config(train_conf_path)
    
    degree = 5

    if "degree" in config.training.task_kwargs:
        degree = config.training.task_kwargs["degree"]
    elif "basis_dim" in config.training.task_kwargs:
        degree = config.training.task_kwargs["basis_dim"]
    else:
        raise ValueError("No degree or basis_dim in config")

    # Set configuration    
    evaluation_kwargs = build_evals(config)
    standard_args = evaluation_kwargs["standard"]
    standard_args["task_sampler_kwargs"] = config.training.task_kwargs

    metrics = {}
    baselines =  get_relevant_baselines_for_degree(degree)
    for model in baselines:
        model.name = model.name[:-len("_driver=None")]
        metrics[model.name] = eval_model(model, include_noise=include_noise, ground_truth_loss=ground_truth_loss, smoothing=smoothing, device=device, **standard_args)

    return metrics
