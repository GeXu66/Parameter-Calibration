import pybamm
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
import csv
import json
import argparse
import multiprocessing
import math
import warnings
import gpytorch
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from joblib import Parallel, delayed
from dataclasses import dataclass
from torch.quasirandom import SobolEngine
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from botorch.fit import fit_gpytorch_mll
from botorch.generation.sampling import ConstrainedMaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize


# Create directories if they don't exist
os.makedirs("./simu_data/Bayes_DC", exist_ok=True)
os.makedirs("./simu_fig/Bayes_DC", exist_ok=True)


def read_dc_data(file_name):
    """Read dynamic current data from CSV file."""
    data = pd.read_csv(file_name)
    # Extract maximum and minimum voltage
    voltage_max = data['V'].max()
    voltage_min = data['V'].min()
    # Extract time information
    time_max = data['time'].values[-1]
    # Extract current profile
    current_profile = data[['time', 'A']].values
    
    # Check if the data has uniform time steps
    time_diffs = np.diff(data['time'].values)
    if not np.allclose(time_diffs, time_diffs[0], rtol=1e-3):
        print("Warning: Non-uniform time steps detected. Using interpolation.")
    
    return time_max, voltage_max, voltage_min, current_profile, 280.0  # Fixed capacity as mentioned in request


def min_max_func(low, high, norm_value):
    """Scale a normalized value between a min and max range."""
    return norm_value * (high - low) + low


def plot_voltage_profile(time, voltage_simulation, voltage_real, rmse_value):
    """Plot the simulated vs real voltage profile."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plotting
    ax.plot(time, voltage_simulation, linestyle='-', label='Simulation')
    ax.plot(time, voltage_real, linestyle='-', label='Experiment')

    plt.xlabel('Time [s]')
    plt.ylabel('Terminal Voltage [V]')
    plt.title(f"Dynamic Current - RMSE: {rmse_value:.4f} V")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"./simu_fig/Bayes_DC/01-T25-DC.png", dpi=300)
    plt.close(fig)


def compute_voltage_profile(sol, file_path, soc_range=(0, 1)):
    """Compute the simulated voltage profile and compare with experimental data."""
    # Extract simulation time and voltage
    time_simulation = sol["Time [s]"].entries
    voltage_simulation = sol["Voltage [V]"].entries

    # Read experimental data
    data = pd.read_csv(file_path)
    time_real = data['time'].values
    voltage_real = data['V'].values
    
    # Process SOC data
    soc_str = data['SOC'].str.replace('%', '', regex=False)
    soc_numeric = pd.to_numeric(soc_str) / 100.0
    
    # Find common time range
    time_max_common = min(time_simulation[-1], time_real[-1])
    
    # Resample to common time points (every second)
    time_resampled = np.arange(0, time_max_common + 1, 1.0)
    
    # Interpolate simulation and experimental data
    interp_sim = interp1d(time_simulation, voltage_simulation, kind='linear', fill_value="extrapolate")
    voltage_simulation_resampled = interp_sim(time_resampled)
    
    interp_real_volt = interp1d(time_real, voltage_real, kind='linear', fill_value="extrapolate")
    voltage_real_resampled = interp_real_volt(time_resampled)
    
    # Interpolate SOC
    interp_real_soc = interp1d(time_real, soc_numeric, kind='linear', fill_value="extrapolate")
    soc_resampled = interp_real_soc(time_resampled)
    
    # Filter by SOC range if specified
    if soc_range == 'all':
        mask = np.ones_like(time_resampled, dtype=bool)
    else:
        low_soc, high_soc = soc_range
        mask = (soc_resampled >= low_soc) & (soc_resampled <= high_soc)
    
    # Check if there are data points in the range
    if not np.any(mask):
        print("Warning: No data points in the specified SOC range.")
        return None, None, None, None, None
    
    # Extract filtered data
    time_filtered = time_resampled[mask]
    voltage_sim_filtered = voltage_simulation_resampled[mask]
    voltage_real_filtered = voltage_real_resampled[mask]
    soc_filtered = soc_resampled[mask]
    
    # Calculate RMSE
    rmse_value = np.sqrt(mean_squared_error(voltage_real_filtered, voltage_sim_filtered))
    print(f"RMSE (SOC range={soc_range}): {rmse_value}")
    
    return time_filtered, voltage_sim_filtered, voltage_real_filtered, soc_filtered, rmse_value


def pybamm_sim_dc(param, min_voltage, max_voltage, current_profile, capacity, temperature=25):
    """Run a PyBaMM simulation with custom current profile."""
    # Map normalized parameters to physical values
    N_parallel = min_max_func(180, 220, param[0])
    electrode_height = min_max_func(0.17, 0.22, param[1])
    electrode_width = min_max_func(0.15, 0.19, param[2])
    Negative_electrode_thickness = min_max_func(80e-6, 120e-6, param[3])
    Positive_electrode_thickness = min_max_func(90e-6, 130e-6, param[4])

    Positive_electrode_active_material_volume_fraction = min_max_func(0.45, 0.6, param[5])
    Negative_electrode_active_material_volume_fraction = min_max_func(0.48, 0.62, param[6])
    Positive_electrode_porosity = min_max_func(0.32, 0.45, param[7])
    Negative_electrode_porosity = min_max_func(0.32, 0.45, param[8])
    Separator_porosity = min_max_func(0.4, 0.6, param[9])

    Positive_electrode_diffusivity = min_max_func(1e-13, 1e-12, param[10])
    Negative_electrode_diffusivity = min_max_func(1e-13, 1e-12, param[11])
    Positive_particle_radius = min_max_func(1e-6, 4e-6, param[12])
    Negative_particle_radius = min_max_func(2e-6, 5e-6, param[13])
    Negative_electrode_conductivity = min_max_func(50.0, 150.0, param[14])
    Positive_electrode_conductivity = min_max_func(30.0, 80.0, param[15])
    Negative_electrode_Bruggeman_coefficient = min_max_func(1.2, 2.0, param[16])
    Positive_electrode_Bruggeman_coefficient = min_max_func(1.2, 2.0, param[17])

    Initial_concentration_in_positive_electrode = min_max_func(25000.0, 32000.0, param[18])
    Initial_concentration_in_negative_electrode = min_max_func(4000.0, 6000.0, param[19])
    Maximum_concentration_in_positive_electrode = min_max_func(45000.0, 58000.0, param[20])
    Maximum_concentration_in_negative_electrode = min_max_func(25000.0, 35000.0, param[21])

    # Set up parameter values
    parameter_values = pybamm.ParameterValues("Prada2013")
    
    # Model options
    options = {"cell geometry": "arbitrary", "thermal": "lumped", "contact resistance": "false"}
    model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
    
    # Create custom current function
    # Current data is in A
    def current_function(t):
        times = current_profile[:, 0]
        currents = current_profile[:, 1]
        # Use linear interpolation for current values
        return np.interp(t, times, currents)
    
    # Create experiment with custom current function
    experiment = pybamm.Experiment(
        [
            (
                "Use custom current function for {} seconds".format(int(current_profile[-1, 0])),
                current_function,
            )
        ],
        period="{}seconds".format(1)  # 1 second timestep as mentioned
    )

    # Set up parameter dictionary
    param_dict = {
        "Number of electrodes connected in parallel to make a cell": N_parallel,
        "Nominal cell capacity [A.h]": capacity,
        "Lower voltage cut-off [V]": min_voltage - 0.1,
        "Upper voltage cut-off [V]": max_voltage + 0.1,
        "Ambient temperature [K]": 273.15 + 25,
        "Initial temperature [K]": 273.15 + temperature,

        # Geometry and structure parameters
        "Electrode height [m]": electrode_height,
        "Electrode width [m]": electrode_width,
        "Negative electrode thickness [m]": Negative_electrode_thickness,
        "Positive electrode thickness [m]": Positive_electrode_thickness,
        "Negative current collector thickness [m]": 15e-6,  # Fixed value
        "Positive current collector thickness [m]": 20e-6,  # Fixed value

        # Material composition parameters
        "Positive electrode active material volume fraction": Positive_electrode_active_material_volume_fraction,
        "Negative electrode active material volume fraction": Negative_electrode_active_material_volume_fraction,
        "Positive electrode porosity": Positive_electrode_porosity,
        "Negative electrode porosity": Negative_electrode_porosity,
        "Separator porosity": Separator_porosity,

        # Transport parameters
        "Positive electrode diffusivity [m2.s-1]": Positive_electrode_diffusivity,
        "Negative electrode diffusivity [m2.s-1]": Negative_electrode_diffusivity,
        "Positive particle radius [m]": Positive_particle_radius,
        "Negative particle radius [m]": Negative_particle_radius,
        "Negative electrode conductivity [S.m-1]": Negative_electrode_conductivity,
        "Positive electrode conductivity [S.m-1]": Positive_electrode_conductivity,
        "Negative electrode Bruggeman coefficient (electrolyte)": Negative_electrode_Bruggeman_coefficient,
        "Positive electrode Bruggeman coefficient (electrolyte)": Positive_electrode_Bruggeman_coefficient,

        # Concentration parameters
        "Initial concentration in positive electrode [mol.m-3]": Initial_concentration_in_positive_electrode,
        "Initial concentration in negative electrode [mol.m-3]": Initial_concentration_in_negative_electrode,
        "Maximum concentration in positive electrode [mol.m-3]": Maximum_concentration_in_positive_electrode,
        "Maximum concentration in negative electrode [mol.m-3]": Maximum_concentration_in_negative_electrode,
    }

    # Update parameter values
    parameter_values.update(param_dict, check_already_exists=False)
    
    # Create solver with appropriate settings for variable current profile
    safe_solver = pybamm.CasadiSolver(mode="safe", dt_max=10)  # Use a smaller dt_max for better accuracy
    
    # Create simulation
    sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=safe_solver, experiment=experiment)
    
    # Run simulation
    sim.solve(initial_soc=1.0)  # Assume starting from full charge
    
    return parameter_values, sim.solution


def main_simulation_dc(param, soc_range, save=False, plot=False):
    """Main simulation function for dynamic current profile."""
    # Read dynamic current data
    file_path = "./bat_data/01#-T25-DC.csv"
    time_max, max_voltage, min_voltage, current_profile, capacity = read_dc_data(file_name=file_path)
    
    # Run simulation
    parameter_values, sol = pybamm_sim_dc(
        param, min_voltage, max_voltage, current_profile, capacity, temperature=25
    )
    
    # Compute voltage profile and RMSE
    time_filtered, voltage_sim_filtered, voltage_real_filtered, soc_filtered, rmse_value = compute_voltage_profile(
        sol=sol, file_path=file_path, soc_range=soc_range
    )
    
    # Save and plot results if requested
    if save and time_filtered is not None:
        df = pd.DataFrame({
            "time": time_filtered, 
            "voltage_real": voltage_real_filtered, 
            "voltage_simulation": voltage_sim_filtered,
            "soc": soc_filtered
        })
        df.to_csv(f"./simu_data/Bayes_DC/01-T25-DC.csv", index=False)
    
    if plot and time_filtered is not None:
        plot_voltage_profile(time_filtered, voltage_sim_filtered, voltage_real_filtered, rmse_value)
    
    return rmse_value


def catch_error_simulation(solution, soc_range, return_dict):
    """Run simulation with error handling."""
    try:
        rmse_value = main_simulation_dc(solution, soc_range)
        return_dict['result'] = rmse_value
        return_dict['reason'] = 'No Problem!'
    except Exception as e:
        print(f"Error occurred: {e}")
        return_dict['result'] = 1.5  # Default high error value on failure
        return_dict['reason'] = str(e)


def run_with_timeout(param, soc_range, timeout=60):
    """Run simulation with timeout protection."""
    param = param.cpu().numpy() if isinstance(param, torch.Tensor) else param
    print('param:', param)
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    process = multiprocessing.Process(target=catch_error_simulation, args=(param, soc_range, return_dict))
    process.start()
    process.join(timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        reason = "Simulation timed out!"
        return 1.5, reason
    
    result = return_dict.get('result')
    try:
        if np.isnan(result):
            reason = "Result is NaN!"
            return 1.5, reason
        else:
            dict_reason = return_dict.get('reason')
            return result, dict_reason
    except Exception as e:
        print(f"Error processing result: {e}")
        reason = str(e)
        return 1.5, reason


def obj_func(solution, soc_range):
    """Objective function for optimization."""
    rmse_value, reason = run_with_timeout(solution, soc_range)
    print("\033[31m Norm Solution Value\033[0m", solution)
    print("\033[31m RMSE (V):\033[0m", rmse_value)
    print("\033[31m Error Reason:\033[0m", reason)
    return rmse_value


def eval_objective(x):
    """Evaluation function for all SOC range."""
    soc_range = 'all'
    return obj_func(x, soc_range)


def eval_c1(x):
    """Constraint for low SOC range."""
    soc_range = (0.05, 0.3)
    return obj_func(x, soc_range) - 0.02  # Constraint: RMSE < 20mV


def eval_c2(x):
    """Constraint for high SOC range."""
    soc_range = (0.7, 0.95)
    return obj_func(x, soc_range) - 0.02  # Constraint: RMSE < 20mV


@dataclass
class ScboState:
    """State class for Constrained Bayesian Optimization."""
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = float("inf")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}
    best_constraint_values: Tensor = torch.ones(2, **tkwargs) * torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.dim) / self.batch_size]))


def update_tr_length(state: ScboState):
    """Update trust region length based on optimization progress."""
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def get_best_index_for_batch(Y: Tensor, C: Tensor):
    """Find the best solution in a batch considering constraints."""
    is_feas = (C <= 0).all(dim=-1)
    if is_feas.any():  # If there are feasible points
        score = Y.clone()
        score[~is_feas] = float("inf")
        return score.argmin()  # Return index of feasible point with lowest objective
    return C.clamp(min=0).sum(dim=-1).argmin()  # Return index of point with least constraint violation


def update_state(state, Y_next, C_next):
    """Update optimization state based on new evaluations."""
    best_ind = get_best_index_for_batch(Y=Y_next, C=C_next)
    y_next, c_next = Y_next[best_ind], C_next[best_ind]

    if (c_next <= 0).all():
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

    state = update_tr_length(state)
    return state


def get_initial_points(dim, n_pts, seed=0):
    """Generate initial points for optimization using Sobol sequence."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}
    
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    X_init = sobol.draw(n=n_pts).to(**tkwargs)
    return X_init


def generate_batch(
        state,
        model,
        X,
        Y,
        C,
        batch_size,
        n_candidates,
        constraint_model,
        sobol: SobolEngine,
):
    """Generate a batch of candidate points using trust region."""
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))

    best_ind = get_best_index_for_batch(Y=Y, C=C)
    x_center = X[best_ind, :].clone()
    tr_lb = torch.clamp(x_center - state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + state.length / 2.0, 0.0, 1.0)

    dim = X.shape[-1]
    pert = sobol.draw(n_candidates).to(**state.tkwargs)
    pert = tr_lb + (tr_ub - tr_lb) * pert

    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, **state.tkwargs) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=state.device)] = 1

    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]

    constrained_thompson_sampling = ConstrainedMaxPosteriorSampling(
        model=model, constraint_model=constraint_model, replacement=False
    )
    with torch.no_grad():
        X_next = constrained_thompson_sampling(X_cand, num_samples=batch_size)

    return X_next


def get_fitted_model(X, Y):
    """Fit and return a Gaussian Process model to the data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}
    
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=X.shape[1], lengthscale_constraint=Interval(0.005, 4.0))
    )
    model = SingleTaskGP(
        X,
        Y,
        covar_module=covar_module,
        likelihood=likelihood,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    with gpytorch.settings.max_cholesky_size(float("inf")):
        fit_gpytorch_mll(mll)

    return model


def save_data(train_X, train_Y, C1, C2, filename='data.json'):
    """Save optimization data to JSON file."""
    # Convert tensors to NumPy arrays
    train_X_np = train_X.cpu().numpy()
    train_Y_np = train_Y.cpu().numpy().flatten()
    C1_np = C1.cpu().numpy().flatten()
    C2_np = C2.cpu().numpy().flatten()

    # Create data list
    data_list = []
    for i in range(train_X_np.shape[0]):
        data_list.append({
            "train_X": train_X_np[i].tolist(),
            "train_Y": float(train_Y_np[i]),
            "C1": float(C1_np[i]),
            "C2": float(C2_np[i])
        })

    # Sort by objective value
    data_list.sort(key=lambda x: x["train_Y"])

    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(data_list, f, indent=4)


def optimize_battery_params():
    """Main optimization function using constrained Bayesian optimization."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    tkwargs = {"device": device, "dtype": dtype}
    dim = 22  # Number of parameters
    batch_size = 4  # Smaller batch size for more frequent updates
    n_init = 10  # Initial points
    stop_length = 100  # Maximum number of evaluations
    
    # Generate initial points
    train_X = get_initial_points(dim, n_init)

    # Evaluate initial points
    print("Evaluating initial points...")
    njobs = min(multiprocessing.cpu_count(), n_init)
    train_Y_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_objective)(x) for x in train_X)
    C1_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_c1)(x) for x in train_X)
    C2_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_c2)(x) for x in train_X)

    # Count valid evaluations
    useful_point = sum(1 for y in train_Y_list if y != 1.5)
    print(f"Number of useful initial points: {useful_point}/{n_init}")

    # Convert to tensors
    train_Y = torch.tensor(train_Y_list, **tkwargs).unsqueeze(-1)
    C1 = torch.tensor(C1_list, **tkwargs).unsqueeze(-1)
    C2 = torch.tensor(C2_list, **tkwargs).unsqueeze(-1)

    # Initialize optimization state
    state = ScboState(dim, batch_size=batch_size)
    N_CANDIDATES = 50  # Number of candidates to generate
    sobol = SobolEngine(dim, scramble=True, seed=1)

    # Lists to store feasible solutions
    feasible_solutions = []
    feasible_rmse = []
    feasible_c1 = []
    feasible_c2 = []

    try:
        # Main optimization loop
        while len(train_Y) < stop_length:
            # Fit GP models
            model = get_fitted_model(train_X, train_Y)
            c1_model = get_fitted_model(train_X, C1)
            c2_model = get_fitted_model(train_X, C2)
            
            # Generate new candidate points
            with gpytorch.settings.max_cholesky_size(float("inf")):
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
            
            # Evaluate new candidates
            Y_next_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_objective)(x) for x in X_next)
            C1_next_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_c1)(x) for x in X_next)
            C2_next_list = Parallel(n_jobs=njobs, prefer="threads")(delayed(eval_c2)(x) for x in X_next)
            
            # Convert to tensors
            Y_next = torch.tensor(Y_next_list, **tkwargs).unsqueeze(-1)
            C1_next = torch.tensor(C1_next_list, **tkwargs).unsqueeze(-1)
            C2_next = torch.tensor(C2_next_list, **tkwargs).unsqueeze(-1)
            
            # Store feasible solutions
            for i in range(len(Y_next_list)):
                if Y_next_list[i] != 1.5:
                    feasible_solutions.append(X_next[i].cpu().numpy())
                    feasible_rmse.append(Y_next_list[i])
                    feasible_c1.append(C1_next_list[i])
                    feasible_c2.append(C2_next_list[i])
                    # Keep only the best 100 solutions
                    if len(feasible_solutions) > 100:
                        worst_idx = np.argmax(feasible_rmse)
                        del feasible_solutions[worst_idx]
                        del feasible_rmse[worst_idx]
                        del feasible_c1[worst_idx]
                        del feasible_c2[worst_idx]
            
            # Update valid point count
            useful_point += sum(1 for y in Y_next_list if y != 1.5)
            print(f"Total useful points: {useful_point}/{len(train_Y) + len(Y_next)}")
            
            # Update optimization state
            C_next = torch.cat([C1_next, C2_next], dim=-1)
            state = update_state(state=state, Y_next=Y_next, C_next=C_next)
            
            # Add new data to training set
            train_X = torch.cat((train_X, X_next), dim=0)
            train_Y = torch.cat((train_Y, Y_next), dim=0)
            C1 = torch.cat((C1, C1_next), dim=0)
            C2 = torch.cat((C2, C2_next), dim=0)
            
            # Print current status
            if (state.best_constraint_values <= 0).all():
                print(f"{len(train_X)}) Best RMSE: {state.best_value:.4e}, TR length: {state.length:.2e}")
            else:
                violation = state.best_constraint_values.clamp(min=0).sum()
                print(f"{len(train_X)}) No feasible point yet! Constraint violation: {violation:.2e}, TR length: {state.length:.2e}")
            
            # Periodically save results
            if len(train_Y) % 10 == 0:
                save_data(train_X, train_Y, C1, C2, filename='./simu_data/Bayes_DC/optimization_progress.json')
                
                # Save best solution so far
                best_ind = get_best_index_for_batch(Y=train_Y, C=torch.cat((C1, C2), dim=-1))
                best_params = train_X[best_ind].cpu().numpy()
                
                # Run simulation with best parameters and save results
                _ = main_simulation_dc(best_params, 'all', save=True, plot=True)
                
                # Save current solutions to CSV
                if feasible_solutions:
                    sorted_data = sorted(zip(feasible_solutions, feasible_rmse, feasible_c1, feasible_c2), key=lambda x: x[1])
                    with open('./simu_data/Bayes_DC/solutions.csv', 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerow(['Solution', 'RMSE', 'C1', 'C2'])
                        for solution, rmse, c1, c2 in sorted_data:
                            writer.writerow([solution, rmse, c1, c2])
    
    except Exception as e:
        print(f"Optimization interrupted: {e}")
    
    finally:
        # Save final results
        save_data(train_X, train_Y, C1, C2, filename='./simu_data/Bayes_DC/final_data.json')
        
        # Get best solution
        best_ind = get_best_index_for_batch(Y=train_Y, C=torch.cat((C1, C2), dim=-1))
        best_params = train_X[best_ind].cpu().numpy()
        
        # Run final simulation with best parameters
        best_rmse = main_simulation_dc(best_params, 'all', save=True, plot=True)
        print(f"Best RMSE: {best_rmse}")
        print(f"Best parameters: {best_params}")
        
        # Plot optimization progress
        fig, ax = plt.subplots(figsize=(10, 6))
        min_values = np.minimum.accumulate(train_Y.cpu().numpy())
        plt.plot(min_values, marker="o", lw=2)
        plt.plot([0, len(train_Y)], [0.02, 0.02], "k--", lw=2)
        plt.ylabel("RMSE (V)", fontsize=14)
        plt.xlabel("Number of evaluations", fontsize=14)
        plt.xlim([0, len(train_Y)])
        plt.grid(True)
        plt.title("Optimization Progress", fontsize=16)
        fig.savefig('./simu_fig/Bayes_DC/optimization_progress.png', dpi=300)
        plt.close(fig)
        
        # Save final solutions to CSV
        if feasible_solutions:
            sorted_data = sorted(zip(feasible_solutions, feasible_rmse, feasible_c1, feasible_c2), key=lambda x: x[1])
            with open('./simu_data/Bayes_DC/final_solutions.csv', 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Solution', 'RMSE', 'C1', 'C2'])
                for solution, rmse, c1, c2 in sorted_data:
                    writer.writerow([solution, rmse, c1, c2])
        
        return best_params, best_rmse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dynamic current battery parameter identification")
    parser.add_argument('--train', action='store_true', default=True, help='Run optimization')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate best solution')
    args = parser.parse_args()
    
    if args.train:
        print("Starting parameter optimization for dynamic current profile...")
        best_params, best_rmse = optimize_battery_params()
        print(f"Optimization complete. Best RMSE: {best_rmse}")
    
    if args.evaluate:
        # Load best solution from CSV if available
        try:
            solutions_df = pd.read_csv('./simu_data/Bayes_DC/final_solutions.csv')
            best_solution_str = solutions_df.iloc[0]['Solution']
            best_solution = np.array(eval(best_solution_str))
            rmse = main_simulation_dc(best_solution, 'all', save=True, plot=True)
            print(f"Evaluated best solution. RMSE: {rmse}")
        except Exception as e:
            print(f"Could not evaluate best solution: {e}")
            print("Run with --train first or provide a valid solution file.") 