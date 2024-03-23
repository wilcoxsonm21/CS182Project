import json
import os
import sys

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml

import models
from samplers import get_data_sampler, sample_transformation
from tasks import get_task_sampler
from models import get_relevant_baselines_for_degree

def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    #This code prints the component of the prompt within the range of read-in projection and the orthogonal component
    if conf.model.family == "gpt2-soft-prompt":
        print(torch.linalg.pinv(model.transformer_model._read_in.weight).shape)
        closestHardPrompt = (model.prompt - model.transformer_model._read_in.bias) @ torch.linalg.pinv(model.transformer_model._read_in.weight.T)
        print("PROMPT: ", closestHardPrompt)
        print("Orthogonal component: ", torch.linalg.norm(model.prompt - model.transformer_model._read_in(closestHardPrompt)))
        print("total norm: ", torch.linalg.norm(model.prompt))
    
    elif conf.model.family == "gpt2-hard-prompt":
        print("PROMPT: ", model.prompt)
        print("total norm: ", torch.linalg.norm(model.prompt))

    return model, conf


# Functions for evaluation


def eval_batch(model, task_sampler, xs, include_noise=True, ground_truth_loss=False, smoothing=0):
    task = task_sampler()
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm", "gpt2-soft-prompt", "gpt2-hard-prompt"]:
        device = "cuda"
    else:
        device = "cpu"
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
        pred = model(cur_xs.to(device), ys.to(device)).detach()
        predictions[i] = pred.cpu()
    predictions = predictions.mean(dim=0)
    if ground_truth_loss:
        metrics = task.get_metric()(predictions, ys - noise)
    else: 
        metrics = task.get_metric()(predictions, ys)
    return metrics

def get_imputed_ys(model, task, xs, ys, test_x, noise=False, smoothing = 0):
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm", "gpt2-soft-prompt",  "gpt2-hard-prompt"]:
        device = "cuda"
    else:
        device = "cpu"
    predictions = []
    next_ys = task.evaluate(test_x, noise=noise)
    for i in range(test_x.shape[1]):
        center = test_x[:, i, :].unsqueeze(0)
        perturbations = np.arange(-1 * smoothing + center, smoothing + center + 0.0002, 0.0002)
        batched_eval = torch.zeros(len(perturbations), xs.shape[1] + 1, xs.shape[2])
        for j in range(len(perturbations)):
            expanded = torch.as_tensor(perturbations[j]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
            expanded = expanded.float()
            batched_eval[j] = torch.cat([xs, expanded], dim=1)
        cur_ys = torch.cat([ys, next_ys[:,i].unsqueeze(0)], dim=1)        
        cur_ys = cur_ys.repeat(len(perturbations), 1, 1).squeeze(1)    
        pred = model(batched_eval.to(device), cur_ys.to(device)).detach()
        predictions.append(pred.cpu().mean(dim=0)[-1])
    result = torch.stack(predictions, dim=0)
    return result

# Functions for generating different kinds of train/test data

def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None


def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
    include_noise=True,
    ground_truth_loss=False,
    smoothing=0,
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """
    assert num_eval_examples % batch_size == 0

    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    all_metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"]
    for i in range(num_eval_examples // batch_size):
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)
        if not isinstance(model, models.TransformerModel):
            metrics = eval_batch(model, task_sampler, xs, include_noise=include_noise, ground_truth_loss=ground_truth_loss, smoothing=0)
        else:
            metrics = eval_batch(model, task_sampler, xs, include_noise=include_noise, ground_truth_loss=ground_truth_loss, smoothing=smoothing)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf):
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
    }

    evaluation_kwargs = {}

    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
    if task_name != "linear_regression":
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs

    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs


def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}

    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                continue

            metrics[model.name] = eval_model(model, **kwargs)
        all_metrics[eval_name] = metrics

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics

def compute_evals_basis(transformer_models, evaluation_kwargs, save_path=None, recompute=False, include_noise=True, ground_truth_loss=False, smoothing=0):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        print("no metrics found")
        all_metrics = {}
    standard_args = evaluation_kwargs["standard"]
    for i in range(4, 12):
        metrics = {}
        baselines =  get_relevant_baselines_for_degree(i)
        baselines += transformer_models
        if "degree-" + str(i) in all_metrics and not recompute:
            metrics = all_metrics["degree-" + str(i)]
        for model in baselines:
            if model.name in metrics and not recompute:
                continue
            standard_args["task_sampler_kwargs"] = {"degree": i,} # TODO: fix this]
            metrics[model.name] = eval_model(model, include_noise=include_noise, ground_truth_loss=ground_truth_loss, smoothing=smoothing, **standard_args)
        all_metrics["degree-" + str(i)] = metrics

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics



def get_run_metrics(
    run_path, run_path_2=None, run_path_3=None, step=-1, cache=True, skip_model_load=False, skip_baselines=False, include_noise=True, ground_truth_loss=False, smoothing=0):
    model, conf = get_model_from_run(run_path, 400000)
    model.name += "_soft_prompt"
    transformer_model = model.cuda().eval()
    evaluation_kwargs = build_evals(conf)

    transformer_models = [transformer_model]
    if run_path_2 is not None:
        model_2, conf_2 = get_model_from_run(run_path_2, 5000000)
        transformer_model_2 = model_2.cuda().eval()
        transformer_models.append(transformer_model_2)
    if run_path_3 is not None:
        model_3, conf_3 = get_model_from_run(run_path_3, step)
        model_3.name += "_0.5_noise"
        transformer_model_3 = model_3.cuda().eval()
        transformer_models.append(transformer_model_3)

    if not cache:
        save_path = None
    elif step == -1:
        if smoothing > 0:
            save_path = os.path.join(run_path, "metrics_smooth_light.json")
        else:
            save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")
    print(save_path)
    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals_basis(transformer_models, evaluation_kwargs, save_path, recompute, include_noise=include_noise, ground_truth_loss=ground_truth_loss, smoothing=smoothing)
    print(all_metrics)
    return all_metrics



def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
            (16, 8): "Transformer-16",
            (24, 16): "Transformer-plus",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    print(name)
    if "ridge" in name:
        return "Chebyshev Ridge " + name.split("_")[2]
    if "cheby" in name:
        return "Chebyshev " + name.split("_")[1]
    if "kernel" in name:
        return "Kernel Least Squares " + name.split("_")[1]
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    print(df.run_name.unique())
    print(df)
    return df

if __name__ == "__main__":
    run_dir = sys.argv[1]
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        print(f"Evaluating task {task}")
        for run_id in tqdm(os.listdir(task_dir)):
            run_path = os.path.join(run_dir, task, run_id)
            metrics = get_run_metrics(run_path)
