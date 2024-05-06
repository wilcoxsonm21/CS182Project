import os

import matplotlib.pyplot as plt
import seaborn as sns

from eval import get_run_metrics, baseline_names, get_model_from_run
from models import build_model

sns.set_theme("notebook", "darkgrid")
palette = sns.color_palette("colorblind")


relevant_model_names = {
    "kernel_linear_regression": [
        "Transformer",
        "Transformer-16",
        "Kernel Least Squares 11",
        "Chebyshev",
        "Chebyshev Ridge",
    ],
    "linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
    ],
    "sparse_linear_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "Averaging",
        "Lasso (alpha=0.01)",
    ],
    "decision_tree": [
        "Transformer",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
        "Greedy Tree Learning",
        "XGBoost",
    ],
    "relu_2nn_regression": [
        "Transformer",
        "Least Squares",
        "3-Nearest Neighbors",
        "2-layer NN, GD",
    ],
}

def get_model_names_for_degree(degree):
    names = ["Transformer",
            "Transformer Curriculum",
        "Chebyshev " + str(degree),
        "Kernel Least Squares " + str(degree),
        "Chebyshev Ridge " + str(degree),]
    if degree < 11:
        names.append("Chebyshev Ridge 11")
    if degree > 1:
        names.append("Chebyshev Ridge 1")

def basic_plot(metrics, trivial=1.0, yhigh_lim=float('inf')):
    fig, ax = plt.subplots(1, 1)

    color = 0
    ax.axhline(trivial, ls="--", color="gray")
    max_val = 0
    for name, vs in metrics.items():
        ax.plot(vs["mean"], "-", label=name, color=palette[color % 10], lw=2)
        max_val = max(max_val, max(vs["mean"]))
        low = vs["bootstrap_low"]
        high = vs["bootstrap_high"]
        ax.fill_between(range(len(low)), low, high, alpha=0.3)
        color += 1
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("mean squared error")
    #ax.set_xlim(-1, len(low) + 0.1)
    ax.set_ylim(-0.05, min(max_val*1.01, yhigh_lim))

    legend = ax.legend(loc="upper left")#, bbox_to_anchor=(1, 1))
    fig.set_size_inches(4, 3)
    for line in legend.get_lines():
        line.set_linewidth(3)

    return fig, ax


def collect_results(run_dir, df, valid_row=None, rename_eval=None, rename_model=None, smoothing=0, step=4000):
    all_metrics = {}
    for _, r in df.iterrows():
        print("Valid row?:", valid_row(r), r.task, r.run_id)
        if valid_row is not None and not valid_row(r):
            continue

        run_path = os.path.join(run_dir, r.task, r.run_id)
        _, conf = get_model_from_run(run_path, only_conf=True)

        print(r.run_name, r.run_id)
        metrics = get_run_metrics(run_path, skip_model_load=True, smoothing=smoothing, step=step)
        print("Metrics:", metrics)
        for eval_name, results in sorted(metrics.items()):
            processed_results = {}
            print(eval_name)
            for model_name, m in results.items():
                if "gpt2" in model_name in model_name:
                    old_model_name = model_name
                    model_name = r.model
                    if "noise" in old_model_name:
                        model_name += " N(0, " + str(old_model_name.split("_")[-2]) + ") Noise"
                    if "soft" in old_model_name:
                        model_name += " Soft Prompt"
                    if "hard" in old_model_name:
                        model_name += " Hard Prompt"
                    if "curriculum" in old_model_name:
                        model_name += " Curriculum"
                    if "batch" in old_model_name:
                        model_name += " Batch Size " + str(old_model_name.split("_")[-2])
                    if rename_model is not None:
                        model_name = rename_model(model_name, r)
                else:
                    model_name = baseline_names(model_name)
                    print(model_name)
                m_processed = {}
                n_dims = conf.model.n_dims

                xlim = conf.model.n_positions #2 * n_dims + 1
                if r.task in ["relu_2nn_regression", "decision_tree"]:
                    xlim = 200

                normalization = n_dims
                if r.task == "sparse_linear_regression":
                    normalization = int(r.kwargs.split("=")[-1])
                if r.task == "decision_tree":
                    normalization = 1

                for k, v in m.items():
                    v = v[:xlim]
                    v = [vv / normalization for vv in v]
                    m_processed[k] = v
                processed_results[model_name] = m_processed
            if rename_eval is not None:
                eval_name = rename_eval(eval_name, r)
            if eval_name not in all_metrics:
                all_metrics[eval_name] = {}
            all_metrics[eval_name].update(processed_results)
    return all_metrics
