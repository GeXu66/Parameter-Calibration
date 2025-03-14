import math
import os
import warnings
from dataclasses import dataclass

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine
import matplotlib.pyplot as plt
import numpy as np
from botorch.fit import fit_gpytorch_mll
# Constrained Max Posterior Sampling s a new sampling class, similar to MaxPosteriorSampling,
# which implements the constrained version of Thompson Sampling described in [1].
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.test_functions import Ackley
from botorch.utils.transforms import unnormalize

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
tkwargs = {"device": device, "dtype": dtype}

SMOKE_TEST = os.environ.get("SMOKE_TEST")


# When evaluating the function, we must first unnormalize the inputs since
# we will use normalized inputs x in the main optimizaiton loop
def eval_objective(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return fun(unnormalize(x, fun.bounds))


def c1(x):  # Equivalent to enforcing that sum(x) <= 0
    return x.sum()


def c2(x):  # Equivalent to enforcing that ||x||_2 <= 5
    return torch.norm(x, p=2) - 5


# We assume c1, c2 have same bounds as the Ackley function above
def eval_c1(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return c1(unnormalize(x, fun.bounds))


def eval_c2(x):
    """This is a helper function we use to unnormalize and evalaute a point"""
    return c2(unnormalize(x, fun.bounds))


@dataclass
class ScboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2, **tkwargs) * torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))


def update_tr_length(state: ScboState):
    # Update the length of the trust region according to
    # success and failure counters
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:  # Restart when trust region becomes too small
        state.restart_triggered = True

    return state


def get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Return the index for the best point."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # Choose best feasible candidate
        score = Y.clone()
        score[~is_feas] = -float("inf")
        return score.argmax()
    return C.clamp(min=0).sum(dim=-1).argmin()


def update_state(state, Y_next, C_next):
    """Method used to update the TuRBO state after each step of optimization.

    Success and failure counters are updated according to the objective values
    (Y_next) and constraint values (C_next) of the batch of candidate points
    evaluated on the optimization step.

    As in the original TuRBO paper, a success is counted whenver any one of the
    new candidate points improves upon the incumbent best point. The key difference
    for SCBO is that we only compare points by their objective values when both points
    are valid (meet all constraints). If exactly one of the two points being compared
    violates a constraint, the other valid point is automatically considered to be better.
    If both points violate some constraints, we compare them inated by their constraint values.
    The better point in this case is the one with minimum total constraint violation
    (the minimum sum of constraint values)"""

    # Pick the best point from the batch
    best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind], C_next[best_ind]

    if (c_next <= 0).all():
        # At least one new candidate is feasible
        improvement_threshold = state.best_value + 1e-3 * math.fabs(state.best_value)
        if y_next > improvement_threshold or (state.best_constraint_values > 0).any():
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1
    else:
        # No new candidate is feasible
        total_violation_next = c_next.clamp(min=0).sum(dim=-1)
        total_violation_center = state.best_constraint_values.clamp(min=0).sum(dim=-1)
        if total_violation_next < total_violation_center:
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = y_next.item()
            state.best_constraint_values = c_next
        else:
            state.success_counter = 0
            state.failure_counter += 1

    # Update the length of the trust region according to the success and failure counters
    state = update_tr_length(state)
    return state


def get_initial_points(dim, n_pts, seed=0):
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(dtype=dtype, device=device)
    return X_init


def generate_batch(
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        C,  # Constraint values
        batch_size,
        n_candidates,  # Number of candidates for Thompson sampling
        constraint_model,
        sobol: SobolEngine,
):
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    # Create the TR bounds
    best_ind = get_best_index_for_batch(Y=Y, C=C)
    x_center = X[best_ind, :].clone()
    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

    # Thompson Sampling w/ Constraints (SCBO)
    dim = X.shape[-1]
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, **tkwargs) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    # Sample on the candidate points using Constrained Max Posterior Sampling
    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model=model, constraint_model=constraint_model, replacement=False
    )
    with torch.no_grad():
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

    return X_next


def get_fitted_model(X, Y):
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
        MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
        fit_gpytorch_mll(mll)

    return model


if __name__ == '__main__':
    # Here we define the example 10D Ackley function
    fun = Ackley(dim=10, negate=True).to(**tkwargs)
    fun.bounds[0, :].fill_(-5)
    fun.bounds[1, :].fill_(10)
    dim = fun.dim
    lb, ub = fun.bounds
    print(dim)
    print(fun.bounds)

    batch_size = 4
    n_init = 10
    max_cholesky_size = float("inf")  # Always use Cholesky

    # Generate initial data
    train_X = get_initial_points(dim, n_init)
    print(train_X)
    train_Y = torch.tensor([eval_objective(x) for x in train_X], **tkwargs).unsqueeze(-1)
    C1 = torch.tensor([eval_c1(x) for x in train_X], **tkwargs).unsqueeze(-1)
    C2 = torch.tensor([eval_c2(x) for x in train_X], **tkwargs).unsqueeze(-1)

    # Initialize TuRBO state
    state = ScboState(dim, batch_size=batch_size)

    # Note: We use 2000 candidates here to make the tutorial run faster.
    # SCBO actually uses min(5000, max(2000, 200 * dim)) candidate points by default.
    N_CANDIDATES = 2000 if not SMOKE_TEST else 4
    sobol = SobolEngine(dim, scramble=True, seed=1)

    while not state.restart_triggered:  # Run until TuRBO converges
        # Fit GP models for objective and constraints
        model = get_fitted_model(train_X, train_Y)
        c1_model = get_fitted_model(train_X, C1)
        c2_model = get_fitted_model(train_X, C2)

        # Generate a batch of candidates
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            X_next = generate_batch(
                state=state,
                model=model,
                X=train_X,
                Y=train_Y,
                C=torch.cat((C1, C2), dim=-1),
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                constraint_model=ModelListGP(c1_model, c2_model),
                sobol=sobol,
            )

        # Evaluate both the objective and constraints for the selected candidaates
        Y_next = torch.tensor([eval_objective(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
        C1_next = torch.tensor([eval_c1(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
        C2_next = torch.tensor([eval_c2(x) for x in X_next], dtype=dtype, device=device).unsqueeze(-1)
        C_next = torch.cat([C1_next, C2_next], dim=-1)

        # Update TuRBO state
        state = update_state(state=state, Y_next=Y_next, C_next=C_next)

        # Append data. Note that we append all data, even points that violate
        # the constraints. This is so our constraint models can learn more
        # about the constraint functions and gain confidence in where violations occur.
        train_X = torch.cat((train_X, X_next), dim=0)
        train_Y = torch.cat((train_Y, Y_next), dim=0)
        C1 = torch.cat((C1, C1_next), dim=0)
        C2 = torch.cat((C2, C2_next), dim=0)

        # Print current status. Note that state.best_value is always the best
        # objective value found so far which meets the constraints, or in the case
        # that no points have been found yet which meet the constraints, it is the
        # objective value of the point with the minimum constraint violation.
        if (state.best_constraint_values <= 0).all():
            print(f"{len(train_X)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}")
        else:
            violation = state.best_constraint_values.clamp(min=0).sum()
            print(
                f"{len(train_X)}) No feasible point yet! Smallest total violation: "
                f"{violation:.2e}, TR length: {state.length:.2e}"
            )

    fig, ax = plt.subplots(figsize=(8, 6))

    score = train_Y.clone()
    # Set infeasible to -inf
    score[~(torch.cat((C1, C2), dim=-1) <= 0).all(dim=-1)] = float("-inf")
    fx = np.maximum.accumulate(score.cpu())
    plt.plot(fx, marker="", lw=3)

    plt.plot([0, len(train_Y)], [fun.optimal_value, fun.optimal_value], "k--", lw=3)
    plt.ylabel("Function value", fontsize=18)
    plt.xlabel("Number of evaluations", fontsize=18)
    plt.title("10D Ackley with 2 outcome constraints", fontsize=20)
    plt.xlim([0, len(train_Y)])
    plt.ylim([-15, 1])

    plt.grid(True)
    plt.show()
