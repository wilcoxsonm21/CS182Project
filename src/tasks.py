import math

import torch
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import numpy as np
from models import *


def generate_synth_data(
    degree,
    length=100,
    samples=1000,
    perturbation=0.1,
    root_offsets=None,
):
    """
    Generate interleaved sequence data for training a GPT model, where the model predicts
    the next value in a sequence of [x1, f(x1), x2, f(x2), ..., xn] as [f(x1), x2, f(x2), ..., f(xn)],
    for a given number of samples, each with a specified sequence length.

    Parameters:
    - degree (int): Degree of the polynomial used for data generation.
    - length (int): Number of x values (and corresponding f(x) values) per sample.
    - samples (int): Total number of samples to generate.
    - perturbation (float): Perturbation applied to polynomial roots.
    - root_offsets (list or np.ndarray, optional): Specific offsets to apply to Chebyshev roots.

    Returns:
    - torch.Tensor: Input sequences tensor of shape (samples, 2*length-1).
    - torch.Tensor: Target sequences tensor of shape (samples, 2*length-1).
    """
    assert length % 2 == 0, "Length must be an even number."

    inputs_list = []
    targets_list = []

    for _ in range(samples):
        roots, scale = construct_polynomial(degree, perturbation, root_offsets)

        # uniform random sample from [-1, 1]
        x_values = np.random.uniform(-1, 1, (length // 2) + 1)
        

        # shuffle x, y pairs
        indices = np.random.permutation((length // 2) + 1)
        x_values = x_values[indices]
        y_values = y_values[indices]

        interleaved = np.empty(length + 1)
        interleaved[0::2] = x_values  # x values at even indices
        interleaved[1::2] = y_values[
            :-1
        ]  # f(x) values at odd indices, except for the last f(x)xs

        inputs = interleaved[:-1]  # All except the last value
        targets = interleaved[1:]  # All except the first value

        inputs_list.append(inputs)
        targets_list.append(targets)

    # Convert lists to tensors with shape (samples, 2*length-1)
    inputs_tensor = torch.tensor(inputs_list, dtype=torch.float32)
    targets_tensor = torch.tensor(targets_list, dtype=torch.float32)

    return inputs_tensor, targets_tensor

def squared_error(ys_pred, ys):
    return (ys - ys_pred).square()


def mean_squared_error(ys_pred, ys):
    return (ys - ys_pred).square().mean()


def accuracy(ys_pred, ys):
    return (ys == ys_pred.sign()).float()


sigmoid = torch.nn.Sigmoid()
bce_loss = torch.nn.BCELoss()


def cross_entropy(ys_pred, ys):
    output = sigmoid(ys_pred)
    target = (ys + 1) / 2
    return bce_loss(output, target)


class Task:
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None):
        self.n_dims = n_dims
        self.b_size = batch_size
        self.pool_dict = pool_dict
        self.seeds = seeds
        assert pool_dict is None or seeds is None

    def evaluate(self, xs):
        raise NotImplementedError

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        raise NotImplementedError

    @staticmethod
    def get_training_metric():
        raise NotImplementedError


def get_task_sampler(
    task_name, n_dims, batch_size, pool_dict=None, num_tasks=None, **kwargs
):
    task_names_to_classes = {
        "linear_regression": LinearRegression,
        "sparse_linear_regression": SparseLinearRegression,
        "linear_classification": LinearClassification,
        "noisy_linear_regression": NoisyLinearRegression,
        "quadratic_regression": QuadraticRegression,
        "relu_2nn_regression": Relu2nnRegression,
        "decision_tree": DecisionTree,
        "kernel_linear_regression": ChebyshevKernelLinearRegression,
        "chebyshev_kernel_linear_regression": ChebyshevKernelLinearRegression,
        "polynomial_shared_roots": PolynomialSharedRoots,
    }

    if task_name in task_names_to_classes:
        task_cls = task_names_to_classes[task_name]
        if num_tasks is not None:
            if pool_dict is not None:
                raise ValueError("Either pool_dict or num_tasks should be None.")
            pool_dict = task_cls.generate_pool_dict(n_dims, num_tasks, **kwargs)
        return lambda **args: task_cls(n_dims, batch_size, pool_dict, **args, **kwargs)
    else:
        print("Unknown task")
        raise NotImplementedError


class LinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(LinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale

        if pool_dict is None and seeds is None:
            self.w_b = torch.randn(self.b_size, self.n_dims, 1)
        elif seeds is not None:
            self.w_b = torch.zeros(self.b_size, self.n_dims, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.w_b[i] = torch.randn(self.n_dims, 1, generator=generator)
        else:
            assert "w" in pool_dict
            indices = torch.randperm(len(pool_dict["w"]))[:batch_size]
            self.w_b = pool_dict["w"][indices]

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        print(w_b.shape)
        print(xs_b.shape)
        print((xs_b @ w_b).shape)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error

class ChebyshevKernelLinearRegression(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, basis_dim=1, different_degrees=False, lowest_degree=1, highest_degree=1, fixed_coeffs=None, curriculum=None):
        """scale: a constant by which to scale the randomly sampled weights."""
        assert basis_dim >= highest_degree, "Basis dimension must be greater than or equal to highest degree"
        super(ChebyshevKernelLinearRegression, self).__init__(n_dims, batch_size, pool_dict, seeds) 
        self.basis_dim = basis_dim 
        self.curriculum = curriculum
        self.highest_degree = highest_degree
        self.diff_poly_degree = different_degrees 
        self.lowest_degree = lowest_degree
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
        combinations = scale*torch.randn(size=(self.b_size, self.basis_dim + 1)) 
        if fixed_coeffs is not None:
            combinations[:, :fixed_coeffs] = 1
        if self.diff_poly_degree:
            mask = torch.ones(combinations.shape[0], combinations.shape[-1], dtype=torch.float32)
            if curriculum:
                self.highest_degree = curriculum.highest_degree
            print("HIGHEST DEGREE: ", self.highest_degree)
            indices = torch.randint(self.lowest_degree, self.highest_degree + 1, (combinations.shape[0], 1))    # Note the dimensions
            self.indices = indices
            mask[torch.arange(0, combinations.shape[-1], dtype=torch.float32).repeat(combinations.shape[0],1) >= indices] = 0
            combinations = torch.mul(combinations, mask)
            self.mask = mask
        self.w_b = (combinations @ self.chebyshev_coeffs).unsqueeze(2) # note if diff poly degree is false, then only generates basis dim degree polynomials 

    def evaluate(self, xs_b, noise=False, separate_noise=False, noise_variance=0.2):
        expanded_basis = torch.zeros(*xs_b.shape[:-1], xs_b.shape[-1]*(self.basis_dim + 1))
        for i in range(self.basis_dim + 1): #we are also adding the constant term
            expanded_basis[..., i*xs_b.shape[-1]:(i+1)*xs_b.shape[-1]] = xs_b**i
        expanded_basis.to(xs_b.device)        
        w_b = self.w_b.to(xs_b.device)
        ys_b = (expanded_basis @ w_b)[:, :, 0]
        if noise and not separate_noise:
            return ys_b + math.sqrt(noise_variance) * torch.randn_like(ys_b)
        elif noise and separate_noise:
            return ys_b, math.sqrt(noise_variance) * torch.randn_like(ys_b)
        else:
            if separate_noise:
                return ys_b, torch.zeros_like(ys_b)
            else:
                return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    def get_training_loss(self):
        return mean_squared_error

class PolynomialSharedRoots(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, basis_dim=1, degree=1, curriculum=None, perturbation=0.1):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(PolynomialSharedRoots, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.basis_dim = basis_dim
        self.curriculum = curriculum
        self.degree = degree
        self.polys, self.scales = self.construct_polynomial(degree, batch_size=batch_size, perturbation=perturbation, num_to_not_perturb=2)


    def construct_polynomial(
            self,
            degree,
            perturbation=0.1,
            batch_size=1, 
            num_to_not_perturb=0,
        ):
        """
        Constructs and scales a polynomial with roots adjusted relative to the positions of Chebyshev polynomial roots.
        Raises an error if the shift is too large, causing roots to go outside [-1, 1] or cross over adjacent roots.

        Parameters:
        - degree (int): Degree of the polynomial.
        - perturbation (float): Maximum perturbation applied if offsets are not specified.
        - root_offsets (np.ndarray, optional): Array of offsets to apply to the Chebyshev roots.

        Returns:
        - roots (np.ndarray): Array of roots of the polynomial.
        - scale (float): Scaling factor for the polynomial.
        """
        k = np.arange(1, degree + 1)
        chebyshev_roots = np.cos((2 * k - 1) * np.pi / (2 * degree))
        chebyshev_roots = np.sort(chebyshev_roots)
        chebyshev_roots = chebyshev_roots[None, :]
        chebyshev_roots = np.repeat(chebyshev_roots, batch_size, axis=0)
        print(chebyshev_roots.shape)
        roots_to_use = chebyshev_roots.copy()
        roots_to_use[:, num_to_not_perturb:] = chebyshev_roots[:, num_to_not_perturb:] + np.random.uniform(
            -perturbation, perturbation, (batch_size, degree - num_to_not_perturb)
        )
        #roots_to_use[0] = max(roots_to_use[0], -1)
        #roots_to_use[-1] = min(roots_to_use[-1], 1)
        #assert np.all(roots_to_use >= -1) and np.all(
        #    roots_to_use <= 1
        #), "Perturbation causes root to go outside the [-1, 1] range."

        # Scale the polynomial to have a maximum absolute value of 1.
        xs = np.linspace(-1, 1, 100)
        polys = []
        for i in range(batch_size):
            poly = np.poly(roots_to_use[i])
            polys.append(poly)
        scales = np.zeros(batch_size)
        for i in range(batch_size):
            scales[i] = np.random.choice([1, -1]) / np.max(np.abs(np.polyval(polys[i], xs)))
        return polys, scales
    
    def evaluate(self, xs_b, noise=False, separate_noise=False, noise_variance=0.2):   
        ys_b = np.zeros((xs_b.shape[0], xs_b.shape[1]))
        for i in range(xs_b.shape[0]): 
            ys_b[i,:] = self.scales[i]*np.squeeze(np.polyval(self.polys[i], xs_b[i]),axis=1)
        ys_b = torch.tensor(ys_b, dtype=torch.float32)
        if noise and not separate_noise:
            return ys_b + math.sqrt(noise_variance) * torch.randn_like(ys_b)
        elif noise and separate_noise:
            return ys_b, math.sqrt(noise_variance) * torch.randn_like(ys_b)
        else:
            if separate_noise:
                return ys_b, torch.zeros_like(ys_b)
            else:
                return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, **kwargs):  # ignore extra args
        return {"w": torch.randn(num_tasks, n_dims, 1)}

    @staticmethod
    def get_metric():
        return squared_error

    def get_training_loss(self):
        return mean_squared_error
        
    
class KernelLinearRegression(LinearRegression):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, scale=1, basis_dim=1): #TODO only supports axis alligned 
        """scale: a constant by which to scale the randomly sampled weights."""
        self.tasks = []
        self.basis_dim = basis_dim
        self.shift = None
        for i in range(self.basis_dim):
            self.tasks.append(LinearRegression(n_dims*(i + 1), batch_size, pool_dict, seeds, scale))
    
    def evaluate(self, xs_b):
        #random = np.random.randint(0, self.basis_dim)
        random = self.basis_dim - 1
        basis_dim = random + 1
        expanded_basis = torch.zeros(*xs_b.shape[:-1], xs_b.shape[-1]*basis_dim)
        for i in range(basis_dim):
            # We want to normalize the input so the output has the same variance indepedent of basis dimension
            # This involves a coefficient that is inverse of variance for each power of x
            # And another coefficient that is inverse of sqrt of variance for total basis dim since variance is additive
            expanded_basis[..., i*xs_b.shape[-1]:(i+1)*xs_b.shape[-1]] = (1/math.sqrt(basis_dim))*(1/math.sqrt(self.getNthDegreeVariance(i + 1)))*(xs_b**(i + 1))
        expanded_basis.to(xs_b.device)
        standard = self.tasks[random].evaluate(expanded_basis) # Note that we are using log to make the scales
        if self.shift is None:
            self.shift = 100*torch.min(standard) - 1e-5
        
        return standard - self.shift
        
    
    # Returns the expectation of X^n where X is a standard normal random variable
    def getNthDegreeExpectation(self, n):
        if n % 2 == 0:
            return math.factorial(n) / (2**(n/2) * math.factorial(n/2))
        else:
            return 0
    
    # Returns the variance of X^n where X is a standard normal random variable
    def getNthDegreeVariance(self, n):
        return self.getNthDegreeExpectation(2*n) - self.getNthDegreeExpectation(n)**2        

class SparseLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        sparsity=3,
        valid_coords=None,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(SparseLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.sparsity = sparsity
        if valid_coords is None:
            valid_coords = n_dims
        assert valid_coords <= n_dims

        for i, w in enumerate(self.w_b):
            mask = torch.ones(n_dims).bool()
            if seeds is None:
                perm = torch.randperm(valid_coords)
            else:
                generator = torch.Generator()
                generator.manual_seed(seeds[i])
                perm = torch.randperm(valid_coords, generator=generator)
            mask[perm[:sparsity]] = False
            w[mask] = 0

    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b = self.scale * (xs_b @ w_b)[:, :, 0]
        return ys_b

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class LinearClassification(LinearRegression):
    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        return ys_b.sign()

    @staticmethod
    def get_metric():
        return accuracy

    @staticmethod
    def get_training_metric():
        return cross_entropy


class NoisyLinearRegression(LinearRegression):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        noise_std=0,
        renormalize_ys=False,
    ):
        """noise_std: standard deviation of noise added to the prediction."""
        super(NoisyLinearRegression, self).__init__(
            n_dims, batch_size, pool_dict, seeds, scale
        )
        self.noise_std = noise_std
        self.renormalize_ys = renormalize_ys

    def evaluate(self, xs_b):
        ys_b = super().evaluate(xs_b)
        ys_b_noisy = ys_b + torch.randn_like(ys_b) * self.noise_std
        if self.renormalize_ys:
            ys_b_noisy = ys_b_noisy * math.sqrt(self.n_dims) / ys_b_noisy.std()

        return ys_b_noisy


class QuadraticRegression(LinearRegression):
    def evaluate(self, xs_b):
        w_b = self.w_b.to(xs_b.device)
        ys_b_quad = ((xs_b**2) @ w_b)[:, :, 0]
        #         ys_b_quad = ys_b_quad * math.sqrt(self.n_dims) / ys_b_quad.std()
        # Renormalize to Linear Regression Scale
        ys_b_quad = ys_b_quad / math.sqrt(3)
        ys_b_quad = self.scale * ys_b_quad
        return ys_b_quad


class Relu2nnRegression(Task):
    def __init__(
        self,
        n_dims,
        batch_size,
        pool_dict=None,
        seeds=None,
        scale=1,
        hidden_layer_size=100,
    ):
        """scale: a constant by which to scale the randomly sampled weights."""
        super(Relu2nnRegression, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.scale = scale
        self.hidden_layer_size = hidden_layer_size

        if pool_dict is None and seeds is None:
            self.W1 = torch.randn(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.randn(self.b_size, hidden_layer_size, 1)
        elif seeds is not None:
            self.W1 = torch.zeros(self.b_size, self.n_dims, hidden_layer_size)
            self.W2 = torch.zeros(self.b_size, hidden_layer_size, 1)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.W1[i] = torch.randn(
                    self.n_dims, hidden_layer_size, generator=generator
                )
                self.W2[i] = torch.randn(hidden_layer_size, 1, generator=generator)
        else:
            assert "W1" in pool_dict and "W2" in pool_dict
            assert len(pool_dict["W1"]) == len(pool_dict["W2"])
            indices = torch.randperm(len(pool_dict["W1"]))[:batch_size]
            self.W1 = pool_dict["W1"][indices]
            self.W2 = pool_dict["W2"][indices]

    def evaluate(self, xs_b):
        W1 = self.W1.to(xs_b.device)
        W2 = self.W2.to(xs_b.device)
        # Renormalize to Linear Regression Scale
        ys_b_nn = (torch.nn.functional.relu(xs_b @ W1) @ W2)[:, :, 0]
        ys_b_nn = ys_b_nn * math.sqrt(2 / self.hidden_layer_size)
        ys_b_nn = self.scale * ys_b_nn
        #         ys_b_nn = ys_b_nn * math.sqrt(self.n_dims) / ys_b_nn.std()
        return ys_b_nn

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        return {
            "W1": torch.randn(num_tasks, n_dims, hidden_layer_size),
            "W2": torch.randn(num_tasks, hidden_layer_size, 1),
        }

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error


class DecisionTree(Task):
    def __init__(self, n_dims, batch_size, pool_dict=None, seeds=None, depth=4):

        super(DecisionTree, self).__init__(n_dims, batch_size, pool_dict, seeds)
        self.depth = depth

        if pool_dict is None:

            # We represent the tree using an array (tensor). Root node is at index 0, its 2 children at index 1 and 2...
            # dt_tensor stores the coordinate used at each node of the decision tree.
            # Only indices corresponding to non-leaf nodes are relevant
            self.dt_tensor = torch.randint(
                low=0, high=n_dims, size=(batch_size, 2 ** (depth + 1) - 1)
            )

            # Target value at the leaf nodes.
            # Only indices corresponding to leaf nodes are relevant.
            self.target_tensor = torch.randn(self.dt_tensor.shape)
        elif seeds is not None:
            self.dt_tensor = torch.zeros(batch_size, 2 ** (depth + 1) - 1)
            self.target_tensor = torch.zeros_like(dt_tensor)
            generator = torch.Generator()
            assert len(seeds) == self.b_size
            for i, seed in enumerate(seeds):
                generator.manual_seed(seed)
                self.dt_tensor[i] = torch.randint(
                    low=0,
                    high=n_dims - 1,
                    size=2 ** (depth + 1) - 1,
                    generator=generator,
                )
                self.target_tensor[i] = torch.randn(
                    self.dt_tensor[i].shape, generator=generator
                )
        else:
            raise NotImplementedError

    def evaluate(self, xs_b):
        dt_tensor = self.dt_tensor.to(xs_b.device)
        target_tensor = self.target_tensor.to(xs_b.device)
        ys_b = torch.zeros(xs_b.shape[0], xs_b.shape[1], device=xs_b.device)
        for i in range(xs_b.shape[0]):
            xs_bool = xs_b[i] > 0
            # If a single decision tree present, use it for all the xs in the batch.
            if self.b_size == 1:
                dt = dt_tensor[0]
                target = target_tensor[0]
            else:
                dt = dt_tensor[i]
                target = target_tensor[i]

            cur_nodes = torch.zeros(xs_b.shape[1], device=xs_b.device).long()
            for j in range(self.depth):
                cur_coords = dt[cur_nodes]
                cur_decisions = xs_bool[torch.arange(xs_bool.shape[0]), cur_coords]
                cur_nodes = 2 * cur_nodes + 1 + cur_decisions

            ys_b[i] = target[cur_nodes]

        return ys_b

    @staticmethod
    def generate_pool_dict(n_dims, num_tasks, hidden_layer_size=4, **kwargs):
        raise NotImplementedError

    @staticmethod
    def get_metric():
        return squared_error

    @staticmethod
    def get_training_metric():
        return mean_squared_error
