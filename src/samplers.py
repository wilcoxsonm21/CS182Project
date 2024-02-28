import math

import torch
from models import *
from matplotlib import pyplot as plt
from tasks import ChebyshevKernelLinearRegression
import numpy as np
from munch import Munch 
import models
import yaml

import os 

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

    return model, conf

def get_imputed_ys(model, task, xs, ys, test_x, noise=False):
    if torch.cuda.is_available() and model.name.split("_")[0] in ["gpt2", "lstm"]:
        device = "cuda"
    else:
        device = "cpu"
    predictions = []
    next_ys = task.evaluate(test_x, noise=noise)
    for i in range(test_x.shape[1]):
        center = test_x[:, i, :].unsqueeze(2)
        print("center: ", center.shape)
        batched_eval = torch.cat([xs, center], dim=1)
        print(ys.shape)
        print(next_ys[:,i].unsqueeze(1).shape)
        cur_ys = torch.cat([ys, next_ys[:,i].unsqueeze(1)], dim=1)        
        pred = model(batched_eval.to(device), cur_ys.to(device)).detach()
        print("pred: ", pred.shape)
        predictions.append(pred[:, -1:].cpu())
    result = torch.stack(predictions, dim=0)
    return result

class DataSampler:
    def __init__(self, n_dims):
        self.n_dims = n_dims

    def sample_xs(self):
        raise NotImplementedError


def get_data_sampler(data_name, n_dims, **kwargs):
    print("args: ", kwargs)
    names_to_classes = {
        "gaussian": UniformSampler, #,
        "prompting": OptimalSampler,
    }
    if data_name in names_to_classes:
        sampler_cls = names_to_classes[data_name]
        return sampler_cls(n_dims, **kwargs)
    else:
        print("Unknown sampler")
        raise NotImplementedError


def sample_transformation(eigenvalues, normalize=False):
    n_dims = len(eigenvalues)
    U, _, _ = torch.linalg.svd(torch.randn(n_dims, n_dims))
    t = U @ torch.diag(eigenvalues) @ torch.transpose(U, 0, 1)
    if normalize:
        norm_subspace = torch.sum(eigenvalues**2)
        t *= math.sqrt(n_dims / norm_subspace)
    return t


class GaussianSampler(DataSampler):
    def __init__(self, n_dims, bias=None, scale=None):
        super().__init__(n_dims)
        self.bias = bias
        self.scale = scale

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        if seeds is None:
            xs_b = torch.randn(b_size, n_points, self.n_dims)
        else:
            xs_b = torch.zeros(b_size, n_points, self.n_dims)
            generator = torch.Generator()
            assert len(seeds) == b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                xs_b[i] = torch.randn(n_points, self.n_dims, generator=generator)
        if self.scale is not None:
            xs_b = xs_b @ self.scale
        if self.bias is not None:
            xs_b += self.bias
        if n_dims_truncated is not None:
            xs_b[:, :, n_dims_truncated:] = 0
        return xs_b #increasing the variance to sample from N(0, 4I), will see if this breaks performance

class UniformSampler(DataSampler):
    def __init__(self, n_dims, start=-1, end=1):
        super().__init__(n_dims)
        self.start = start
        self.end = end

    def sample_xs(self, n_points, b_size, n_dims_truncated=None, seeds=None):
        xs_b = 2*torch.rand(b_size, n_points, self.n_dims) - 1
        assert torch.min(xs_b) >= -1 and torch.max(xs_b) <= 1
        return xs_b #increasing the variance to sample from N(0, 4I), will see if this breaks performance

class OptimalSampler(DataSampler):
    def __init__(self, n_dims, run_dir, start=-1, end=1):
        super().__init__(n_dims)
        self.start = start
        self.end = end
        self.transformer_model, _ = get_model_from_run(run_dir, -1)
    
    def sample_xs(self, n_points, b_size, task, n_dims_truncated=None, seeds=None):
        sampler = UniformSampler(n_dims=1)
        transformer_model = self.transformer_model.cuda().eval()
        more_xs_for_graphing_truth = sampler.sample_xs(100, b_size)
        extra_indices = np.argsort(more_xs_for_graphing_truth, axis=1)
        extra_ys = task.evaluate(more_xs_for_graphing_truth, noise=False)
        print("extra_ys: ", extra_ys.shape)
        print("extra_indices: ", extra_indices.squeeze().shape)
        print(extra_indices.squeeze()[0])
        extra_ys = extra_ys.gather(1, extra_indices.squeeze())
        #model = ChebyshevKernelLeastSquaresModelWithRidge(basis_dim=6, ridge=0.5)
        #model_no_ridge = ChebyshevKernelLeastSquaresModelWithRidge(basis_dim=9, ridge=0.5)
        #model_low_degree = ChebyshevKernelLeastSquaresModelWithRidge(basis_dim=3, ridge=0.5)
        def plot_performance(xs, ys):
            extra_transformer_estimated = get_imputed_ys(transformer_model, task, xs, ys, more_xs_for_graphing_truth, noise=True).swapaxes(0, 1)
            #extra_predicted = model.return_trained_model(xs, ys)(more_xs_for_graphing_truth)
            #no_ridge_predicted = model_no_ridge.return_trained_model(xs, ys)(more_xs_for_graphing_truth)
            #low_degree_predicted = model_low_degree.return_trained_model(xs, ys)(more_xs_for_graphing_truth)
            #print("low degree predicted: ", low_degree_predicted.shape)
            #print("extra predicted: ", extra_predicted.shape)
            #print("extra indices", extra_indices.shape)
            #print("extra transformer estimated: ", extra_transformer_estimated.shape)
            #low_degree_predicted = low_degree_predicted.gather(1, extra_indices)
            #extra_predicted = extra_predicted.gather(1, extra_indices)
            extra_transformer_estimated = extra_transformer_estimated.gather(1, extra_indices.expand(-1, -1, extra_transformer_estimated.shape[2]))
            #no_ridge_predicted = no_ridge_predicted.gather(1, extra_indices)
            graphing_x = more_xs_for_graphing_truth.gather(1, extra_indices.expand(-1, -1, more_xs_for_graphing_truth.shape[2])).squeeze()
            print("Graphing x: ", graphing_x.shape)
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            #plt.title('Learned Functions after ' + str(i) + ' Points' + ' index 1')
            #ax.scatter(xs[1], ys[1], color='red', label='Noisy Data',zorder=3000)
            #ax.plot(graphing_x[1], extra_transformer_estimated[1], color='purple', label='Transformer')
            #ax.plot(graphing_x[1], extra_predicted[1], color='green', label='Chebyshev Regresson (Degree 9)', zorder=1000)
            #ax.plot(graphing_x[1], extra_ys[1], color='blue', label='Ground Truth',zorder=2000)
            #ax.plot(graphing_x[1], low_degree_predicted[1], color='orange', label='Chebyshev Regresson (Degree 3)', zorder=500)
            #ax.plot(graphing_x[1], no_ridge_predicted[1], color='orange', label='Chebyshev Regresson (Degree 9)')
            #x.set_xlabel('X')
            #ax.set_ylabel('Y')
            #ax.legend()
            #plt.savefig('polynomials/chebyshev_kernel_regression_' + str(i) + ' index 1'  + '.png')
            #plt.show()
            #fig = plt.figure()
            #ax = fig.add_subplot(111)
            #plt.title('Learned Functions after ' + str(i) + ' Points' + ' index 5')
            #ax.scatter(xs[5], ys[5], color='red', label='Noisy Data',zorder=3000)
            #ax.plot(graphing_x[5], extra_transformer_estimated[5], color='purple', label='Transformer')
            #ax.plot(graphing_x[5], extra_predicted[5], color='green', label='Chebyshev Regresson (Degree 9)', zorder=1000)
            #ax.plot(graphing_x[5], extra_ys[5], color='blue', label='Ground Truth',zorder=2000)
            #ax.plot(graphing_x[5], low_degree_predicted[5], color='orange', label='Chebyshev Regresson (Degree 3)', zorder=500)
            #ax.plot(graphing_x[5], no_ridge_predicted[5], color='orange', label='Chebyshev Regresson (Degree 9)')
            #ax.set_xlabel('X')
            #ax.set_ylabel('Y')
            #ax.legend()
            #plt.savefig('polynomials/chebyshev_kernel_regression_' + str(i) + ' index 5'  + '.png')
            #plt.show()
            return extra_transformer_estimated

        def find_x_value_with_largest_error(ground_truth, predicted, xs):
            print("ground truth: ", ground_truth.shape)
            print("predicted: ", predicted.shape)
            print("xs: ", xs.shape)
            
            graphing_x = xs.gather(1, extra_indices)
            errors = np.abs(ground_truth.unsqueeze(2) - predicted)
            max_error_index = np.argmax(errors, axis=1)
            return graphing_x.gather(1, max_error_index.unsqueeze(2).expand(-1, -1, graphing_x.shape[2]))
            
        all_xs = sampler.sample_xs(1, b_size=b_size)
        for i in range(1, n_points - 1, 1):
            all_ys = task.evaluate(all_xs, noise=False)
            learned_model = plot_performance(all_xs, all_ys)
            largest_error_x = find_x_value_with_largest_error(extra_ys, learned_model, more_xs_for_graphing_truth)
            print("xs shape: ", all_xs.shape)
            print("x largest error: ", largest_error_x.shape)
            print("largest error at " + str(i) + " points: " + str(largest_error_x))
            all_xs = torch.cat([all_xs, largest_error_x], axis=1)
        return all_xs
