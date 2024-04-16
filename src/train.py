import os
from pathlib import Path
from random import randint
from typing import List
import uuid
import hydra
import argparse

from quinine import Quinfig, QuinineArgumentParser
from tqdm import tqdm
import torch
import yaml

from eval import get_run_metrics
from tasks import get_task_sampler
from samplers import get_data_sampler
from curriculum import Curriculum
from schema import schema
from models import build_model

import model_utils
from types import SimpleNamespace

import wandb


from box import Box

torch.backends.cudnn.benchmark = True


def train_step(model, xs, ys, optimizer, task):
    optimizer.zero_grad()
    output = model(xs, ys)
    l = task.get_training_loss()
    loss = l(output, ys)
    loss.backward()
    optimizer.step()
    return loss.detach().item(), output.detach()


def sample_seeds(total_seeds, count):
    seeds = set()
    while len(seeds) < count:
        seeds.add(randint(0, total_seeds - 1))
    return seeds


def train(model, args, device="cuda"):

    noise = args.training.noise_variance != 0

    optimizer = torch.optim.Adam(model.parameters(), lr=args.training.learning_rate)
    curriculum = Curriculum(args.training.curriculum)

    starting_step = 0
    state_path = os.path.join(args.out_dir, "state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        starting_step = state["train_step"]
        for i in range(state["train_step"] + 1):
            curriculum.update()

    n_dims = model.n_dims
    bsize = args.training.batch_size
    data_sampler = get_data_sampler(args.training.data, n_dims=n_dims)
    task_sampler = get_task_sampler(
        args.training.task,
        n_dims,
        bsize,
        num_tasks=args.training.num_tasks,
        curriculum=curriculum,
        **args.training.task_kwargs,
    )
    pbar = tqdm(range(starting_step, args.training.train_steps))

    num_training_examples = args.training.num_training_examples

    for i in pbar:
        data_sampler_args = {}
        task_sampler_args = {}

        if "sparse" in args.training.task:
            task_sampler_args["valid_coords"] = curriculum.n_dims_truncated
        if num_training_examples is not None:
            assert num_training_examples >= bsize
            seeds = sample_seeds(num_training_examples, bsize)
            data_sampler_args["seeds"] = seeds
            task_sampler_args["seeds"] = [s + 1 for s in seeds]

        xs = data_sampler.sample_xs(
            curriculum.n_points,
            bsize,
            curriculum.n_dims_truncated,
            **data_sampler_args,
        )
        task = task_sampler(**task_sampler_args)
        ys = task.evaluate(xs, noise=noise, noise_variance=args.training.noise_variance)


        loss, output = train_step(model, xs.to(device), ys.to(device), optimizer, task)

        point_wise_tags = list(range(curriculum.n_points))
        point_wise_loss_func = task.get_metric()
        point_wise_loss = point_wise_loss_func(output, ys.to(device)).mean(dim=0)

        baseline_loss = (
            sum(
                max(curriculum.n_dims_truncated - ii, 0)
                for ii in range(curriculum.n_points)
            )
            / curriculum.n_points
        )

        if i % args.wandb.log_every_steps == 0:
            wandb.log(
                {
                    "overall_loss": loss,
                    "excess_loss": loss / baseline_loss,
                    "pointwise/loss": dict(
                        zip(point_wise_tags, point_wise_loss.cpu().numpy())
                    ),
                    "n_points": curriculum.n_points,
                    "n_dims": curriculum.n_dims_truncated,
                },
                step=i,
            )

        curriculum.update()

        pbar.set_description(f"loss {loss}")
        if i % args.training.save_every_steps == 0:
            training_state = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_step": i,
            }
            torch.save(training_state, state_path)

        if (
            args.training.keep_every_steps > 0
            and i % args.training.keep_every_steps == 0
            and i > 0
        ):
            torch.save(model.state_dict(), os.path.join(args.out_dir, f"model_{i}.pt"))


def main(args, device="cuda", wandb_mode="online"):
    wandb.init(
        dir=args.out_dir,
        project=args.wandb.project,
        entity=args.wandb.entity,
        config=args.__dict__,
        notes=args.wandb.notes,
        name=args.wandb.name,
        resume=True,
        mode=wandb_mode,
    )

    model = build_model(args.model, device=device)
    model.to(device)
    model.train()

    train(model, args, device=device)

    wandb.finish()

    #if not args.test_run:
    #    _ = get_run_metrics(args.out_dir, step=4000, include_noise=False)  # precompute metrics for eval


def load_and_train(conf: dict, device="cuda", wandb_mode="online") -> str:

    conf, out_dir = model_utils.conf_prepare_output_dir(conf)
    main(conf, device=device, wandb_mode=wandb_mode)

    return out_dir

def train_mulitple_soft_prompts(base_model_dir: Path, prompt_conf: Box, soft_prompt_dims: List, device="cuda", wandb_mode="online"):

    for prompt_dim in soft_prompt_dims:

        prompt_conf_copy = prompt_conf.copy()

        prompt_conf_copy.model.prompt_dim = prompt_dim
        prompt_conf_copy.wandb.name = "prompt_dim_" + str(prompt_dim)
        prompt_conf_copy.training.resume_id = "prompt_dim_" + str(prompt_dim)
        prompt_conf_copy.model.pretrained_model_dir = str(base_model_dir)

        prompt_dir = load_and_train(prompt_conf_copy, device=device, wandb_mode=wandb_mode)

        print(f"Finished training prompt model with prompt dim {prompt_dim}, and saved to {prompt_dir}")


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_mode = "disabled" # online, offline, disabled

    #Try me
    prompt_conf = model_utils.load_config("conf/big_prompting_shared_outside.yaml")
    load_and_train(prompt_conf, device=device, wandb_mode=wandb_mode)
    #base_model_dir = Path("../models/kernel_linear_regression/bigger_model")

    # Train special back with 50 in middle
    #prompt_conf = model_utils.load_config("conf/prompting_small_back.yaml")
    #base_model_dir = Path("../models/kernel_linear_regression/bigger_model")
    #train_mulitple_soft_prompts(base_model_dir, prompt_conf, [11, 21, 31, 41], device=device, wandb_mode=wandb_mode)

    # Train model on soft_prompts in front
    """prompt_conf = model_utils.load_config("conf/prompting_front.yaml")
    base_model_dir = Path("../models/kernel_linear_regression/bigger_model")
    train_mulitple_soft_prompts(base_model_dir, prompt_conf, [50, 60, 70, 80, 90, 100], device=device, wandb_mode=wandb_mode)

    # Train model on soft_prompts in back
    prompt_conf = model_utils.load_config("conf/prompting_back.yaml")
    base_model_dir = Path("../models/kernel_linear_regression/bigger_model")
    train_mulitple_soft_prompts(base_model_dir, prompt_conf, [49, 59, 69, 79, 89, 99], device=device, wandb_mode=wandb_mode)

    # Train model without positional encoding and wothout noise, do soft prompting afterwards
    base_conf = model_utils.load_config("conf/base_model_nopos_nonoise.yaml")
    prompt_conf = model_utils.load_config("conf/prompting_nopos_nonoise.yaml")

    base_model_dir = load_and_train(base_conf, device=device, wandb_mode=wandb_mode)
    print(f"Finished training base model, and saved to {base_model_dir}")

    train_mulitple_soft_prompts(base_model_dir, prompt_conf, [50, 60, 70, 80, 90, 100], device=device, wandb_mode=wandb_mode)

    # Train model with positional encoding and noise, do soft prompting afterwards
    base_conf = model_utils.load_config("conf/base_model.yaml")
    prompt_conf = model_utils.load_config("conf/prompting_noise.yaml")

    base_model_dir = load_and_train(base_conf, device=device, wandb_mode=wandb_mode)
    print(f"Finished training base model, and saved to {base_model_dir}")

    train_mulitple_soft_prompts(base_model_dir, prompt_conf, [50, 60, 70, 80, 90, 100], device=device, wandb_mode=wandb_mode)"""


    
