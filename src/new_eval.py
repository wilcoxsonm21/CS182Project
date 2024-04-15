import torch
from munch import Munch
from pathlib import Path
import yaml
from typing import List, NamedTuple
import numpy as np

import models
from eval import eval_model, build_evals, get_relevant_baselines_for_degree
from tasks import get_task_sampler

import matplotlib.pyplot as plt
import seaborn as sns

from box import Box

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


class LoadInfo(NamedTuple):
    path: Path
    step: int
    alternative_train_conf_path: Path = None
    alternative_name: str = ""
    name_addon: str = ""

def get_config(config_path: Path) -> Box:

    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        return Box(yaml.safe_load(fp))
    

def eval_batch_att(model, task_sampler, xs, device="cuda"):

    # Generat random x-values between -1 and 1
    

    task = task_sampler()
    assert include_noise == False
    perturbations = np.arange(-1 * smoothing, smoothing + 0.002, 0.002)
    predictions = torch.zeros(len(perturbations), xs.shape[0], xs.shape[1])
    if ground_truth_loss:
        ys, noise = task.evaluate(xs, noise=include_noise, separate_noise=True)
        ys = ys + noise
    else:
        ys = task.evaluate(xs, noise=include_noise, separate_noise=False)
    for i in range(len(perturbations)):
        cur_xs = xs + perturbations[i]
        #print(device)
        pred = model.attattention_matrix(cur_xs.to(device), ys.to(device)).detach()
        predictions[i] = pred.cpu()
    predictions = predictions.mean(dim=0)
    if ground_truth_loss:
        metrics = task.get_metric()(predictions, ys - noise)
    else: 
        metrics = task.get_metric()(predictions, ys)
    return metrics

def new_get_model_from_run(run_path: Path, step, device="cuda") -> torch.nn.Module:

    config_path = run_path / "config.yaml"
    print(config_path)
    conf = get_config(config_path)

    model = models.build_model(conf.model, device=device)

    if step == -1:
        state_path = run_path / "state.pt"
        state = torch.load(state_path, map_location=torch.device(device))
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = run_path / f"model_{step}.pt"
        state_dict = torch.load(model_path, map_location=torch.device(device))
        model.load_state_dict(state_dict)

    return model, conf


def new_get_run_metrics2(run_path: Path, step: int, device: str = "cuda", alternative_train_conf_path=None):

    # Get model
    model, config = new_get_model_from_run(run_path, step, device=device)
    model = model.eval()

    #print(model)

    if alternative_train_conf_path is not None:
        config.training = get_config(alternative_train_conf_path).training

    # Set configuration    
    evaluation_kwargs = build_evals(config)
    standard_args = evaluation_kwargs["standard"]
    standard_args["task_sampler_kwargs"] = config.training.task_kwargs

    # Loop wanted degrees
    gen_task_sampler = get_task_sampler(
        standard_args["task_name"], standard_args["n_dims"], standard_args["batch_size"], **standard_args["task_sampler_kwargs"]
    )

    #print(evaluation_kwargs, standard_args)

    task_sampler = gen_task_sampler()
    xs = torch.rand((config.training.batch_size, config.training.curriculum.points.end, 1)) * 2 - 1
    ys = task_sampler.evaluate(xs)

    vals, attention = model.attention_matrix(xs.to(device), ys.to(device))

    return vals, attention


def new_get_run_metrics(run_path: Path, step: int, device: str = "cuda", include_noise=True, ground_truth_loss=False, smoothing=0, alternative_train_conf_path=None):

    # Get model
    model, config = new_get_model_from_run(run_path, step, device=device)
    model = model.eval()

    if alternative_train_conf_path is not None:
        config.training = get_config(alternative_train_conf_path).training

    # Set configuration    
    evaluation_kwargs = build_evals(config)
    standard_args = evaluation_kwargs["standard"]
    standard_args["task_sampler_kwargs"] = config.training.task_kwargs

    # Loop wanted degrees
    metrics = eval_model(model, include_noise=include_noise, ground_truth_loss=ground_truth_loss, smoothing=smoothing, device=device, **standard_args)
    task_sampler = get_task_sampler(
        **standard_args
    )

    return metrics

def baseline_data(train_conf_path: Path, include_noise=True, ground_truth_loss=False, smoothing=0, device="cuda"):

    config = get_config(train_conf_path)
    degree = config.training.task_kwargs["degree"]

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

def check_dict_values(source_dict, filter_dict):
    """
    Recursively checks if all values in the filter_dict have the same values in the source_dict.
    Returns True if all values match, False otherwise.
    """
    for key, filter_value in filter_dict.items():
        if key not in source_dict:
            return False
        source_value = source_dict[key]
        if isinstance(filter_value, dict) and isinstance(source_value, dict):
            # If both values are dictionaries, recurse
            if not check_dict_values(source_value, filter_value):
                return False
        elif source_value != filter_value:
            # If values don't match, return False
            return False
    return True

def get_modelpaths_by_filter(models_parent_directories: List[Path], config_filter: dict) -> List[Path]:

    model_paths = []
    for models_parent_directory in models_parent_directories:
        for model_path in models_parent_directory.iterdir():
            if model_path.is_dir():
                config_path = model_path / "config.yaml"
                if config_path.exists():
                    config = get_config(config_path)
                    if check_dict_values(config, config_filter):
                        model_paths.append(model_path)

                else:
                    print(f"Warning: {config_path} not found.")

    return model_paths

if __name__ == "__main__":
    pass