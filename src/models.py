import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, Lasso
import warnings
from sklearn import tree
import xgboost as xgb
import math
import yaml
from munch import Munch
import os
from peft import LoraConfig, get_peft_model

from base_models import NeuralNetwork, ParallelNetworks

class EmptyLayer(nn.Module):

    def __init__(self, device="cuda"):
        super().__init__()
        self.zero = torch.tensor([0], requires_grad=False).to(device)

    def forward(self, x):
        return self.zero

def get_model_from_run(run_path, step=-1, only_conf=False, device="cuda"):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = build_model(conf.model, device=device)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path, torch.device(device))
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path, torch.device(device))
        model.load_state_dict(state_dict)
    return model, conf

def build_model(conf, device="cuda"):

    model = None

    #print("Build model:", device)

    if conf.family == "gpt2":
        model = TransformerModel(
            n_dims=conf.n_dims,
            n_positions=conf.n_positions,
            n_embd=conf.n_embd,
            n_layer=conf.n_layer,
            n_head=conf.n_head,
        )

    elif conf.family == "gpt2-soft-prompt":
        #if "steps" in conf:
        #    model, _ = get_model_from_run(conf.pretrained_model_dir, step=conf.steps, device=device)
        #   print("SoftPrompt Non-tranaible parameters:", model.get_non_trainable_params())
        #    print("SoftPrompt Trainable parameters:", model.get_trainable_params(), "\n")
        #else:    
        transformer_model, _ = get_model_from_run(conf.pretrained_model_dir, device=device)
        model = SoftPromptTransformerModel(transformer_model, conf)
        print("SoftPrompt Non-tranaible parameters:", model.get_non_trainable_params())
        print("SoftPrompt Trainable parameters:", model.get_trainable_params(), "\n")
    elif conf.family == "gpt2-hard-prompt":
        transformer_model, _ = get_model_from_run(conf.pretrained_model_dir, device=device)
        model = HardPromptTransformerModel(transformer_model, conf)
    elif conf.family == "gpt2-lora":
        transformer_model, _ = get_model_from_run(conf.pretrained_model_dir, device=device)
        model = LoraTransformerModel(transformer_model, lora_config=LoraConfig(**conf.lora_config))
        print("Lora Non-tranaible parameters:", model.get_non_trainable_params())
        print("Lora Trainable parameters:", model.get_trainable_params(), "\n")
        #print(model)
    else:
        raise NotImplementedError

    print("Device:", device)
    if conf.family == "gpt2" and not conf.positional_encodings:
        print("NO POSITIONAL ENCODINGS!!!!!!!!!!!!!!!!!!!!!")
        model._backbone.wte = EmptyLayer(device=device)
        model._backbone.wpe = EmptyLayer(device=device)

    elif not conf.positional_encodings:
        print("NO POSITIONAL ENCODINGS!!!!!!!!!!!!!!!!!!!!!")
        model.transformer_model._backbone.wte = EmptyLayer(device=device)
        model.transformer_model._backbone.wpe = EmptyLayer(device=device)

    return model


def get_relevant_baselines(task_name):
    task_to_baselines = {
        "kernel_linear_regression": [
            (KernelLeastSquaresModel, {"basis_dim": 11}), #TODO: Avoid hard coding
            (ChebyshevKernelLeastSquaresModel, {"basis_dim": 11}),
            (ChebyshevKernelLeastSquaresModelWithRidge, {"basis_dim": 11}),
        ],
        "linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "linear_classification": [
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ],
        "sparse_linear_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
        ]
        + [(LassoModel, {"alpha": alpha}) for alpha in [1, 0.1, 0.01, 0.001, 0.0001]],
        "relu_2nn_regression": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (AveragingModel, {}),
            (
                GDModel,
                {
                    "model_class": NeuralNetwork,
                    "model_class_args": {
                        "in_size": 20,
                        "hidden_size": 100,
                        "out_size": 1,
                    },
                    "opt_alg": "adam",
                    "batch_size": 100,
                    "lr": 5e-3,
                    "num_steps": 100,
                },
            ),
        ],
        "decision_tree": [
            (LeastSquaresModel, {}),
            (NNModel, {"n_neighbors": 3}),
            (DecisionTreeModel, {"max_depth": 4}),
            (DecisionTreeModel, {"max_depth": None}),
            (XGBoostModel, {}),
            (AveragingModel, {}),
        ],
    }

    models = [model_cls(**kwargs) for model_cls, kwargs in task_to_baselines[task_name]]
    return models

def get_relevant_baselines_for_degree(degree):
    task_for_degree =  [
        (ChebyshevKernelLeastSquaresModel, {"basis_dim": degree}),
        (ChebyshevKernelLeastSquaresModelWithRidge, {"basis_dim": degree}),
        ]

    models = [model_cls(**kwargs) for model_cls, kwargs in task_for_degree]
    return models


class TransformerModel(nn.Module):
    def __init__(self, n_dims, n_positions, n_embd=128, n_layer=12, n_head=4):
        super(TransformerModel, self).__init__()
        configuration = GPT2Config(
            n_positions=2 * n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            use_cache=False,
        )
        self.name = f"gpt2_embd={n_embd}_layer={n_layer}_head={n_head}"

        self.n_positions = n_positions
        self.n_dims = n_dims
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.prompt_dim = 0
        self._read_in = nn.Linear(n_dims, n_embd)
        self._backbone = GPT2Model(configuration)
        self._read_out = nn.Linear(n_embd, 1)

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)
        return zs

    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)
        embeds = self._read_in(zs)
        output = self._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self._read_out(output)
        return prediction[:, ::2, 0][:, inds]  # predict only on xs
    

class LoraTransformerModel(nn.Module):

    def __init__(self, transformer_model: TransformerModel, lora_config: LoraConfig):
        super(LoraTransformerModel, self).__init__()
        
        self.transformer_model = transformer_model

        self.n_positions, self.n_dims, self.n_embd = transformer_model.n_positions, transformer_model.n_dims, transformer_model.n_embd 
        self.n_head, self.n_layer, self.prompt_dim = transformer_model.n_head, transformer_model.n_layer, transformer_model.prompt_dim

        # Freeze original model
        for param in self.parameters():
            param.requires_grad = False        

        self.lora_config = lora_config
        self.transformer_model._backbone = get_peft_model(self.transformer_model._backbone, self.lora_config)
        self.name = f"gpt2-lora_embd={self.transformer_model.n_embd}_layer={self.transformer_model.n_layer}_head={self.transformer_model.n_head}"

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_non_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
    
    @staticmethod
    def _combine(xs_b, ys_b):
        return TransformerModel._combine(xs_b, ys_b)
    
    def forward(self, xs, ys, inds=None):
        return self.transformer_model(xs, ys, inds)
    

class SoftPromptTransformerModel(nn.Module):
    def __init__(self, transformer_model, conf):
        super(SoftPromptTransformerModel, self).__init__()
        self.transformer_model = transformer_model
        self.prompt_dim = conf.prompt_dim
        self.n_positions = conf.n_positions
        self.n_dims = conf.n_dims
        self.n_embd = self.transformer_model.n_embd
        self.n_layer = self.transformer_model.n_layer
        self.n_head = self.transformer_model.n_head
        for param in self.transformer_model.parameters():
            param.requires_grad = False
        start_inputs = torch.randn((1,self.prompt_dim*2,1))
        self.prompt = nn.Parameter(self.transformer_model._read_in(start_inputs)) # batch size, prompt dim * 2 (x and y), embedding dim, initialize with a possible actual input
        self.name = f"gpt2-soft-prompt_embd={self.n_embd}_layer={self.n_layer}_head={self.n_head}"
    
    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self.transformer_model._combine(xs, ys)
        embeds = self.transformer_model._read_in(zs)
        prompt = self.prompt.repeat(xs.shape[0], 1, 1)
        embeds = torch.cat((prompt, embeds), dim=1)
        output = self.transformer_model._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self.transformer_model._read_out(output)
        return prediction[:, self.prompt_dim*2::2, 0][:, inds]  # predict only on xs, and only after the prompt
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_non_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)
    

class HardPromptTransformerModel(nn.Module):
    def __init__(self, transformer_model, conf):
        super(HardPromptTransformerModel, self).__init__()
        self.transformer_model = transformer_model
        self.prompt_dim = conf.prompt_dim
        self.n_positions = conf.n_positions
        self.n_dims = conf.n_dims
        self.n_embd = self.transformer_model.n_embd
        self.n_layer = self.transformer_model.n_layer
        self.n_head = self.transformer_model.n_head
        for param in self.transformer_model.parameters():
            param.requires_grad = False
        degree = 5 #TODO: Avoid hard coding
        import numpy as np
        k = np.arange(1, degree + 1)
        chebyshev_roots = np.cos((2 * k - 1) * np.pi / (2 * degree))
        chebyshev_roots = np.sort(chebyshev_roots)
        start_inputs = torch.tensor([[chebyshev_roots[0]], [0], [chebyshev_roots[1]], [0]]).type(torch.FloatTensor)
        print(start_inputs)
        self.prompt = nn.Parameter(start_inputs) # batch size, prompt dim * 2 (x and y), 1, initialize with a possible actual input (this is a hard prompt now)
        self.name = f"gpt2-hard-prompt_embd={self.n_embd}_layer={self.n_layer}_head={self.n_head}"
    
    def forward(self, xs, ys, inds=None):
        if inds is None:
            inds = torch.arange(ys.shape[1])
        else:
            inds = torch.tensor(inds)
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")
        zs = self.transformer_model._combine(xs, ys)
        zs = torch.cat((self.prompt.repeat(xs.shape[0], 1, 1), zs), dim=1)
        embeds = self.transformer_model._read_in(zs)
        output = self.transformer_model._backbone(inputs_embeds=embeds).last_hidden_state
        prediction = self.transformer_model._read_out(output)
        return prediction[:, self.prompt_dim*2::2, 0][:, inds]  # predict only on xs, and only after the prompt


class NNModel:
    def __init__(self, n_neighbors, weights="uniform"):

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.name = f"NN_n={n_neighbors}_{weights}"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            dist = (train_xs - test_x).square().sum(dim=2).sqrt()

            if self.weights == "uniform":
                weights = torch.ones_like(dist)
            else:
                weights = 1.0 / dist
                inf_mask = torch.isinf(weights).float()  # deal with exact match
                inf_row = torch.any(inf_mask, axis=1)
                weights[inf_row] = inf_mask[inf_row]

            pred = []
            k = min(i, self.n_neighbors)
            ranks = dist.argsort()[:, :k]
            for y, w, n in zip(train_ys, weights, ranks):
                y, w = y[n], w[n]
                pred.append((w * y).sum() / w.sum())
            preds.append(torch.stack(pred))

        return torch.stack(preds, dim=1)


# xs and ys should be on cpu for this method. Otherwise the output maybe off in case when train_xs is not full rank due to the implementation of torch.linalg.lstsq.
class LeastSquaresModel:
    def __init__(self, driver=None):
        self.driver = driver
        self.name = f"OLS_driver={driver}"

    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            ws, _, _, _ = torch.linalg.lstsq(
                train_xs, train_ys.unsqueeze(2), driver=self.driver
            )

            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)

class KernelLeastSquaresModel(LeastSquaresModel):
    def __init__(self, basis_dim:int = 1, driver=None, random=True):
        super().__init__(driver)
        self.basis_dim = basis_dim
        self.name = f"kernel_{basis_dim}_driver={driver}"
        self.random = random
    
    def __call__(self, xs, ys, inds=None): #TODO: Not sure what inds do, need to probably fix later
        expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
        for i in range(self.basis_dim + 1): #we are also adding the constant term
            # We want to normalize the input so the output has the same variance indepedent of basis dimension
            # This involves a coefficient that is inverse of variance for each power of x
            # And another coefficient that is inverse of sqrt of variance for total basis dim since variance is additive
            expanded_basis[..., i*xs.shape[-1]:(i+1)*xs.shape[-1]] = xs**i
        return super().__call__(expanded_basis, ys, inds)

class ChebyshevKernelLeastSquaresModel(LeastSquaresModel):
    def __init__(self, basis_dim:int = 1, driver = None, random=True):
        super().__init__(driver)
        self.basis_dim = basis_dim
        self.name = f"chebyshev_{basis_dim}_driver={driver}"
        self.random = random
    
    def __call__(self, xs, ys, inds=None):
        expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
        self.chebyshev_coeffs = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, -8, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, -20, 0, 16, 0, 0, 0, 0, 0, 0],
            [-1, 0, 18, 0, -48, 0, 32, 0, 0, 0, 0, 0],
            [0, -7, 0, 56, 0, -112, 0, 64, 0, 0, 0, 0],
            [1, 0, -32, 0, 160, 0, -256, 0, 128, 0, 0, 0],
            [0, 9, 0, -120, 0, 432, 0, -576, 0, 256, 0, 0],
            [-1, 0, 50, 0, -400, 0, 1120, 0, -1280, 0, 512, 0],
            [0, -11, 0, 220, 0, -1232, 0, 2816, 0, -2816, 0, 1024]
        ], dtype=torch.float)
        self.chebyshev_coeffs = self.chebyshev_coeffs[:self.basis_dim + 1, :self.basis_dim + 1]
        expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
        for i in range(self.basis_dim + 1): #we are also adding the constant term
            expanded_basis[..., i*xs.shape[-1]:(i+1)*xs.shape[-1]] = xs**i
        expanded_basis = expanded_basis @ self.chebyshev_coeffs.T
        return super().__call__(expanded_basis, ys, inds)
    
    def return_trained_model(self, xs, ys):
        expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
        self.chebyshev_coeffs = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, -8, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, -20, 0, 16, 0, 0, 0, 0, 0, 0],
            [-1, 0, 18, 0, -48, 0, 32, 0, 0, 0, 0, 0],
            [0, -7, 0, 56, 0, -112, 0, 64, 0, 0, 0, 0],
            [1, 0, -32, 0, 160, 0, -256, 0, 128, 0, 0, 0],
            [0, 9, 0, -120, 0, 432, 0, -576, 0, 256, 0, 0],
            [-1, 0, 50, 0, -400, 0, 1120, 0, -1280, 0, 512, 0],
            [0, -11, 0, 220, 0, -1232, 0, 2816, 0, -2816, 0, 1024]
        ], dtype=torch.float)
        self.chebyshev_coeffs = self.chebyshev_coeffs[:self.basis_dim + 1, :self.basis_dim + 1]
        expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
        for i in range(self.basis_dim + 1): #we are also adding the constant term
            expanded_basis[..., i*xs.shape[-1]:(i+1)*xs.shape[-1]] = xs**i
        expanded_basis = expanded_basis @ self.chebyshev_coeffs.T
        xs, ys = expanded_basis.cpu(), ys.cpu()
        A = torch.bmm(xs.transpose(1, 2), xs)
        C = A
        D = torch.linalg.inv(C)
        E = torch.bmm(D, xs.transpose(1, 2))
        ws = torch.bmm(E, ys.unsqueeze(2))
        def predict(xs):
            expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
            for i in range(self.basis_dim + 1):
                expanded_basis[..., i*xs.shape[-1]:(i+1)*xs.shape[-1]] = xs**i
            expanded_basis = expanded_basis @ self.chebyshev_coeffs.T
            return expanded_basis @ ws
        return predict
    
class ChebyshevKernelLeastSquaresModelWithRidge(LeastSquaresModel):
    def __init__(self, basis_dim:int = 1, ridge = 0.2, driver = None, random=True):
        super().__init__(driver)
        self.basis_dim = basis_dim
        self.name = f"ridge_chebyshev_{basis_dim}_driver={driver}"
        self.random = random
        self.ridge = ridge
        #print(self.ridge)
    
    def __call__(self, xs, ys, inds=None):
        expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
        self.chebyshev_coeffs = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, -8, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, -20, 0, 16, 0, 0, 0, 0, 0, 0],
            [-1, 0, 18, 0, -48, 0, 32, 0, 0, 0, 0, 0],
            [0, -7, 0, 56, 0, -112, 0, 64, 0, 0, 0, 0],
            [1, 0, -32, 0, 160, 0, -256, 0, 128, 0, 0, 0],
            [0, 9, 0, -120, 0, 432, 0, -576, 0, 256, 0, 0],
            [-1, 0, 50, 0, -400, 0, 1120, 0, -1280, 0, 512, 0],
            [0, -11, 0, 220, 0, -1232, 0, 2816, 0, -2816, 0, 1024]
        ], dtype=torch.float)
        self.chebyshev_coeffs = self.chebyshev_coeffs[:self.basis_dim + 1, :self.basis_dim + 1]
        expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
        for i in range(self.basis_dim + 1): #we are also adding the constant term
            expanded_basis[..., i*xs.shape[-1]:(i+1)*xs.shape[-1]] = xs**i
        expanded_basis = expanded_basis @ self.chebyshev_coeffs.T
        xs, ys = expanded_basis.cpu(), ys.cpu()
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []
        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]
            A = torch.bmm(train_xs.transpose(1, 2), train_xs)
            B = self.ridge*torch.eye(train_xs.shape[2]).unsqueeze(0).repeat(train_xs.shape[0], 1, 1)
            C = A + B
            D = torch.linalg.inv(C)
            E = torch.bmm(D, train_xs.transpose(1, 2))
            ws = torch.bmm(E, train_ys.unsqueeze(2))
            pred = test_x @ ws
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)    
    
    def return_trained_model(self, xs, ys):
        expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
        self.chebyshev_coeffs = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -3, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, -8, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, -20, 0, 16, 0, 0, 0, 0, 0, 0],
            [-1, 0, 18, 0, -48, 0, 32, 0, 0, 0, 0, 0],
            [0, -7, 0, 56, 0, -112, 0, 64, 0, 0, 0, 0],
            [1, 0, -32, 0, 160, 0, -256, 0, 128, 0, 0, 0],
            [0, 9, 0, -120, 0, 432, 0, -576, 0, 256, 0, 0],
            [-1, 0, 50, 0, -400, 0, 1120, 0, -1280, 0, 512, 0],
            [0, -11, 0, 220, 0, -1232, 0, 2816, 0, -2816, 0, 1024]
        ], dtype=torch.float)
        self.chebyshev_coeffs = self.chebyshev_coeffs[:self.basis_dim + 1, :self.basis_dim + 1]
        expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
        for i in range(self.basis_dim + 1): #we are also adding the constant term
            expanded_basis[..., i*xs.shape[-1]:(i+1)*xs.shape[-1]] = xs**i
        expanded_basis = expanded_basis @ self.chebyshev_coeffs.T
        xs, ys = expanded_basis.cpu(), ys.cpu()
        A = torch.bmm(xs.transpose(1, 2), xs)
        B = self.ridge*torch.eye(xs.shape[2]).unsqueeze(0).repeat(xs.shape[0], 1, 1)
        C = A + B
        D = torch.linalg.inv(C)
        E = torch.bmm(D, xs.transpose(1, 2))
        ws = torch.bmm(E, ys.unsqueeze(2))
        def predict(xs):
            expanded_basis = torch.zeros(*xs.shape[:-1], xs.shape[-1]*(self.basis_dim + 1))
            for i in range(self.basis_dim + 1):
                expanded_basis[..., i*xs.shape[-1]:(i+1)*xs.shape[-1]] = xs**i
            expanded_basis = expanded_basis @ self.chebyshev_coeffs.T
            return expanded_basis @ ws
        return predict
    
class AveragingModel:
    def __init__(self):
        self.name = "averaging"

    def __call__(self, xs, ys, inds=None):
        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        for i in inds:
            if i == 0:
                preds.append(torch.zeros_like(ys[:, 0]))  # predict zero for first point
                continue
            train_xs, train_ys = xs[:, :i], ys[:, :i]
            test_x = xs[:, i : i + 1]

            train_zs = train_xs * train_ys.unsqueeze(dim=-1)
            w_p = train_zs.mean(dim=1).unsqueeze(dim=-1)
            pred = test_x @ w_p
            preds.append(pred[:, 0, 0])

        return torch.stack(preds, dim=1)


# Lasso regression (for sparse linear regression).
# Seems to take more time as we decrease alpha.
class LassoModel:
    def __init__(self, alpha, max_iter=100000):
        # the l1 regularizer gets multiplied by alpha.
        self.alpha = alpha
        self.max_iter = max_iter
        self.name = f"lasso_alpha={alpha}_max_iter={max_iter}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    # If all points till now have the same label, predict that label.

                    clf = Lasso(
                        alpha=self.alpha, fit_intercept=False, max_iter=self.max_iter
                    )

                    # Check for convergence.
                    with warnings.catch_warnings():
                        warnings.filterwarnings("error")
                        try:
                            clf.fit(train_xs, train_ys)
                        except Warning:
                            print(f"lasso convergence warning at i={i}, j={j}.")
                            raise

                    w_pred = torch.from_numpy(clf.coef_).unsqueeze(1)

                    test_x = xs[j, i : i + 1]
                    y_pred = (test_x @ w_pred.float()).squeeze(1)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


# Gradient Descent and variants.
# Example usage: gd_model = GDModel(NeuralNetwork, {'in_size': 50, 'hidden_size':400, 'out_size' :1}, opt_alg = 'adam', batch_size = 100, lr = 5e-3, num_steps = 200)
class GDModel:
    def __init__(
        self,
        model_class,
        model_class_args,
        opt_alg="sgd",
        batch_size=1,
        num_steps=1000,
        lr=1e-3,
        loss_name="squared",
    ):
        # model_class: torch.nn model class
        # model_class_args: a dict containing arguments for model_class
        # opt_alg can be 'sgd' or 'adam'
        # verbose: whether to print the progress or not
        # batch_size: batch size for sgd
        self.model_class = model_class
        self.model_class_args = model_class_args
        self.opt_alg = opt_alg
        self.lr = lr
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.loss_name = loss_name

        self.name = f"gd_model_class={model_class}_model_class_args={model_class_args}_opt_alg={opt_alg}_lr={lr}_batch_size={batch_size}_num_steps={num_steps}_loss_name={loss_name}"

    def __call__(self, xs, ys, inds=None, verbose=False, print_step=100):
        # inds is a list containing indices where we want the prediction.
        # prediction made at all indices by default.
        # xs: bsize X npoints X ndim.
        # ys: bsize X npoints.
        xs, ys = xs.cuda(), ys.cuda()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []  # predict one for first point

        # i: loop over num_points
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            model = ParallelNetworks(
                ys.shape[0], self.model_class, **self.model_class_args
            )
            model.cuda()
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])

                train_xs, train_ys = xs[:, :i], ys[:, :i]
                test_xs, test_ys = xs[:, i : i + 1], ys[:, i : i + 1]

                if self.opt_alg == "sgd":
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr)
                elif self.opt_alg == "adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
                else:
                    raise NotImplementedError(f"{self.opt_alg} not implemented.")

                if self.loss_name == "squared":
                    loss_criterion = nn.MSELoss()
                else:
                    raise NotImplementedError(f"{self.loss_name} not implemented.")

                # Training loop
                for j in range(self.num_steps):

                    # Prepare batch
                    mask = torch.zeros(i).bool()
                    perm = torch.randperm(i)
                    mask[perm[: self.batch_size]] = True
                    train_xs_cur, train_ys_cur = train_xs[:, mask, :], train_ys[:, mask]

                    if verbose and j % print_step == 0:
                        model.eval()
                        with torch.no_grad():
                            outputs = model(train_xs_cur)
                            loss = loss_criterion(
                                outputs[:, :, 0], train_ys_cur
                            ).detach()
                            outputs_test = model(test_xs)
                            test_loss = loss_criterion(
                                outputs_test[:, :, 0], test_ys
                            ).detach()
                            print(
                                f"ind:{i},step:{j}, train_loss:{loss.item()}, test_loss:{test_loss.item()}"
                            )

                    optimizer.zero_grad()

                    model.train()
                    outputs = model(train_xs_cur)
                    loss = loss_criterion(outputs[:, :, 0], train_ys_cur)
                    loss.backward()
                    optimizer.step()

                model.eval()
                pred = model(test_xs).detach()

                assert pred.shape[1] == 1 and pred.shape[2] == 1
                pred = pred[:, 0, 0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class DecisionTreeModel:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.name = f"decision_tree_max_depth={max_depth}"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in inds:
            pred = torch.zeros_like(ys[:, 0])

            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)
                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0]

            preds.append(pred)

        return torch.stack(preds, dim=1)


class XGBoostModel:
    def __init__(self):
        self.name = "xgboost"

    # inds is a list containing indices where we want the prediction.
    # prediction made at all indices by default.
    def __call__(self, xs, ys, inds=None):
        xs, ys = xs.cpu(), ys.cpu()

        if inds is None:
            inds = range(ys.shape[1])
        else:
            if max(inds) >= ys.shape[1] or min(inds) < 0:
                raise ValueError("inds contain indices where xs and ys are not defined")

        preds = []

        # i: loop over num_points
        # j: loop over bsize
        for i in tqdm(inds):
            pred = torch.zeros_like(ys[:, 0])
            if i > 0:
                pred = torch.zeros_like(ys[:, 0])
                for j in range(ys.shape[0]):
                    train_xs, train_ys = xs[j, :i], ys[j, :i]

                    clf = xgb.XGBRegressor()

                    clf = clf.fit(train_xs, train_ys)
                    test_x = xs[j, i : i + 1]
                    y_pred = clf.predict(test_x)
                    pred[j] = y_pred[0].item()

            preds.append(pred)

        return torch.stack(preds, dim=1)
