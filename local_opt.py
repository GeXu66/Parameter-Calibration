import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import cyipopt
import argparse
import multiprocessing
import pybamm
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import os
import json
from joblib import Parallel, delayed


def read_csv_solution(csv_file):
    """
    Read the first solution from the CSV file produced by Bayesian optimization.

    Args:
        csv_file (str): Path to the CSV file containing Bayesian optimization results

    Returns:
        np.ndarray: The best parameter set from Bayesian optimization
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Get the first row solution (best solution)
        solution_str = df.iloc[0]['Solution']
        cleaned_str = solution_str.strip('[]')  # 去除可能的方括号和两端空白
        numbers = cleaned_str.split()  # 按空格分割成列表
        solution = np.array(list(map(float, numbers)))  # 转为浮点数后创建数组

        print(solution)
        print(f"Successfully loaded solution from {csv_file}")
        print(f"Solution shape: {solution.shape}")
        print(f"Solution: {solution}")

        return solution
    except Exception as e:
        print(f"Error reading CSV solution: {e}")
        raise


def min_max_func(low, high, norm_value):
    """
    Scale a normalized value (0-1) to a specified range.

    Args:
        low (float): Lower bound of the range
        high (float): Upper bound of the range
        norm_value (float): Normalized value between 0 and 1

    Returns:
        float: Scaled value between low and high
    """
    return norm_value * (high - low) + low


def read_file(file_name):
    """
    Read battery data file and extract key parameters.

    Args:
        file_name (str): Path to battery data CSV file

    Returns:
        tuple: time_max, voltage_max, voltage_min, capacity
    """
    data = pd.read_csv(file_name)
    time_max = data['time'].values[-1]
    voltage_max = data['V'].values[0]
    voltage_min = data['V'].values[-1] - 1
    capacity = data['Ah'].values[-1]
    return time_max, voltage_max, voltage_min, capacity


def pybamm_sim(param, min_voltage, max_voltage, discharge_cur, time_max, capacity, temperature, file):
    """
    Run a PyBAMM simulation with the given parameters.

    Args:
        param (np.ndarray): Array of normalized battery parameters
        min_voltage (float): Minimum voltage cutoff
        max_voltage (float): Maximum voltage cutoff
        discharge_cur (float): Discharge current rate
        time_max (float): Maximum simulation time
        capacity (float): Battery capacity
        temperature (float): Operating temperature
        file (str): Path to battery data file

    Returns:
        tuple: parameter_values, solution
    """
    # Map normalized parameters to physical parameters
    electrode_height = min_max_func(0.6, 1, param[0])
    electrode_width = min_max_func(25, 30, param[1])
    Negative_electrode_conductivity = min_max_func(14, 215, param[2])
    Positive_electrode_diffusivity = min_max_func(5.9e-18, 1e-14, param[3])
    Positive_particle_radius = min_max_func(1e-8, 1e-5, param[4])
    Initial_concentration_in_positive_electrode = min_max_func(35.3766672, 31513, param[5])
    Initial_concentration_in_negative_electrode = min_max_func(48.8682, 29866, param[6])
    Positive_electrode_conductivity = min_max_func(0.18, 100, param[7])
    Negative_particle_radius = min_max_func(0.0000005083, 0.0000137, param[8])
    Negative_electrode_thickness = min_max_func(0.000036, 0.0007, param[9])
    Total_heat_transfer_coefficient = min_max_func(5, 35, param[10])
    Separator_density = min_max_func(397, 2470, param[11])
    Separator_thermal_conductivity = min_max_func(0.10672, 0.34, param[12])
    Positive_electrode_porosity = min_max_func(0.12728395, 0.4, param[13])
    Separator_specific_heat_capacity = min_max_func(700, 1978, param[14])
    Maximum_concentration_in_positive_electrode = min_max_func(22806, 63104, param[15])
    Negative_electrode_Bruggeman_coefficient = min_max_func(1.5, 4, param[16])
    Positive_electrode_Bruggeman_coefficient = min_max_func(1.5, 4, param[17])
    Separator_porosity = min_max_func(0.39, 1, param[18])
    Negative_current_collector_thickness = min_max_func(0.00001, 0.000025, param[19])
    Positive_current_collector_thickness = min_max_func(0.00001, 0.000025, param[20])
    Positive_electrode_thickness = min_max_func(0.000042, 0.0001, param[21])
    Positive_electrode_active_material_volume_fraction = min_max_func(0.28485556, 0.665, param[22])
    Negative_electrode_specific_heat_capacity = min_max_func(700, 1437, param[23])
    Positive_electrode_thermal_conductivity = min_max_func(1.04, 2.1, param[24])
    Negative_electrode_active_material_volume_fraction = min_max_func(0.372403, 0.75, param[25])

    # Create parameter dictionary for PyBAMM
    parameter_values = pybamm.ParameterValues("Prada2013")
    option = {"cell geometry": "arbitrary", "thermal": "lumped", "contact resistance": "false"}

    # Select model type
    if model_type == "DFN":
        model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
    else:
        model = pybamm.lithium_ion.SPM()  # Single Particle Model

    # Setup experiment
    exp = pybamm.Experiment(
        [(f"Discharge at {discharge_cur} C for {time_max} seconds",)]
    )

    # Define parameter dictionary with all battery parameters
    param_dict = {
        "Number of electrodes connected in parallel to make a cell": 1,
        "Nominal cell capacity [A.h]": capacity,
        "Lower voltage cut-off [V]": min_voltage - 0.5,
        "Upper voltage cut-off [V]": max_voltage + 0.5,
        "Ambient temperature [K]": 273.15 + 25,
        "Initial temperature [K]": 273.15 + temperature,
        # Cell parameters
        "Electrode height [m]": electrode_height,
        "Electrode width [m]": electrode_width,
        "Negative electrode conductivity [S.m-1]": Negative_electrode_conductivity,
        "Positive electrode diffusivity [m2.s-1]": Positive_electrode_diffusivity,
        "Positive particle radius [m]": Positive_particle_radius,
        "Initial concentration in positive electrode [mol.m-3]": Initial_concentration_in_positive_electrode,
        "Initial concentration in negative electrode [mol.m-3]": Initial_concentration_in_negative_electrode,
        "Positive electrode conductivity [S.m-1]": Positive_electrode_conductivity,
        "Negative particle radius [m]": Negative_particle_radius,
        "Negative electrode thickness [m]": Negative_electrode_thickness,
        "Total heat transfer coefficient [W.m-2.K-1]": Total_heat_transfer_coefficient,
        "Separator density [kg.m-3]": Separator_density,
        "Separator thermal conductivity [W.m-1.K-1]": Separator_thermal_conductivity,
        "Positive electrode porosity": Positive_electrode_porosity,
        "Separator specific heat capacity [J.kg-1.K-1]": Separator_specific_heat_capacity,
        "Maximum concentration in positive electrode [mol.m-3]": Maximum_concentration_in_positive_electrode,
        "Negative electrode Bruggeman coefficient (electrolyte)": Negative_electrode_Bruggeman_coefficient,
        "Positive electrode Bruggeman coefficient (electrolyte)": Positive_electrode_Bruggeman_coefficient,
        "Separator porosity": Separator_porosity,
        "Negative current collector thickness [m]": Negative_current_collector_thickness,
        "Positive current collector thickness [m]": Positive_current_collector_thickness,
        "Positive electrode thickness [m]": Positive_electrode_thickness,
        "Positive electrode active material volume fraction": Positive_electrode_active_material_volume_fraction,
        "Negative electrode specific heat capacity [J.kg-1.K-1]": Negative_electrode_specific_heat_capacity,
        "Positive electrode thermal conductivity [W.m-1.K-1]": Positive_electrode_thermal_conductivity,
        "Negative electrode active material volume fraction": Negative_electrode_active_material_volume_fraction,
    }

    # Update parameter values
    parameter_values.update(param_dict, check_already_exists=False)

    # Define solver and create simulation
    safe_solver = pybamm.CasadiSolver(mode="safe", dt_max=120)
    sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=safe_solver, experiment=exp)

    # Run simulation
    sim.solve(initial_soc=1)
    sol = sim.solution

    return parameter_values, sol


def compute_time_discharge(sol, file_path, soc_range=(0.9, 1)):
    """
    Compute RMSE between simulation and experimental discharge curves.

    Args:
        sol (pybamm.Solution): PyBAMM solution object
        file_path (str): Path to experimental data file
        soc_range (tuple): SOC range to consider for RMSE calculation

    Returns:
        tuple: time_resampled_out, voltage_sim_filtered, voltage_real_filtered, soc_resampled_out, rmse_value
    """
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
    time_max_real = time_real[-1]
    time_max_sim = time_simulation[-1]
    real_time_max = min(time_max_real, time_max_sim)

    # Resample time (every 10 seconds)
    time_resampled = np.arange(0, real_time_max + 1, 10)

    # Interpolate simulation and experimental voltage
    interp_func_sim = interp1d(time_simulation, voltage_simulation, kind='linear', fill_value="extrapolate")
    voltage_simulation_resampled = interp_func_sim(time_resampled)

    interp_func_real_volt = interp1d(time_real, voltage_real, kind='linear', fill_value="extrapolate")
    voltage_real_resampled = interp_func_real_volt(time_resampled)

    # Interpolate SOC
    interp_func_real_soc = interp1d(time_real, soc_numeric, kind='linear', fill_value="extrapolate")
    soc_resampled = interp_func_real_soc(time_resampled)

    # Filter by SOC range
    if soc_range == 'all':
        # Use all data points
        mask = np.ones_like(time_resampled, dtype=bool)
    else:
        # Filter by SOC range
        low_soc, high_soc = soc_range
        mask = (soc_resampled >= low_soc) & (soc_resampled <= high_soc)

    # Check if any data points remain after filtering
    if not np.any(mask):
        print("Warning: No data points in the specified SOC range.")
        return None, None, None, None, None

    # Extract filtered data
    time_resampled_out = time_resampled[mask]
    voltage_sim_filtered = voltage_simulation_resampled[mask]
    voltage_real_filtered = voltage_real_resampled[mask]
    soc_resampled_out = soc_resampled[mask]

    # Calculate RMSE
    rmse_value = np.sqrt(mean_squared_error(voltage_real_filtered, voltage_sim_filtered))
    print(f"TIME DISCHARGE RMSE (SOC range={soc_range}): {rmse_value}")

    return time_resampled_out, voltage_sim_filtered, voltage_real_filtered, soc_resampled_out, rmse_value


def main_simulation(param, soc_range, save=False, plot=False):
    """
    Run simulation for all battery test conditions and compute RMSE.

    Args:
        param (np.ndarray): Battery parameter array
        soc_range (str or tuple): SOC range for RMSE calculation
        save (bool): Whether to save simulation results
        plot (bool): Whether to plot results

    Returns:
        list: RMSE values for all test conditions
    """
    # Set experiment names and process them
    names = name.split(",")
    file_list = [f"./bat_data/{single}.csv" for single in names]
    all_time_rmse = []

    for i, file in enumerate(file_list):
        # Extract test conditions from filename
        discharge_cur = float(names[i].split("-")[-1].replace("C", ""))
        temperature = int(names[i].split("-")[1].replace("T", ""))

        # Read battery data
        time_max, max_voltage, min_voltage, capacity = read_file(file_name=file)

        # Run simulation
        parameter_values, sol = pybamm_sim(param, min_voltage, max_voltage, discharge_cur, time_max, capacity, temperature, file)

        # Compute RMSE
        time_resampled_out, voltage_sim_filtered, voltage_real_filtered, soc_resampled_out, rmse_value = compute_time_discharge(
            sol=sol, file_path=file, soc_range=soc_range
        )

        all_time_rmse.append(rmse_value)

        # Save simulation results if requested
        if save and rmse_value is not None:
            # Ensure directories exist
            os.makedirs(f"./simu_data/{subdir_name}", exist_ok=True)

            df = pd.DataFrame({
                "real_time": time_resampled_out,
                "real_voltage": voltage_real_filtered,
                "simu_time": time_resampled_out,
                "simu_voltage": voltage_sim_filtered
            })

            # Extract the file name base from the original path
            file_name_base = os.path.basename(file).split('.')[0]
            df.to_csv(f"./simu_data/{subdir_name}/exp_{file_name_base}-T{temperature}-{discharge_cur}C-{model_type}.csv",
                      index=False, sep=",")

        # Plot results if requested
        if plot and rmse_value is not None:
            # Ensure directories exist
            os.makedirs(f"./simu_fig/{subdir_name}", exist_ok=True)

            fig, ax = plt.subplots()
            ax.plot(time_resampled_out, voltage_sim_filtered, linestyle='-', label='Simulation')
            ax.plot(time_resampled_out, voltage_real_filtered, linestyle='-', label='Experiment')

            plt.xlabel('Time [s]')
            plt.ylabel('Terminal Voltage [V]')

            # Extract the file name base from the original path
            file_name_base = os.path.basename(file).split('.')[0]
            plt.title(f"{file_name_base}-{discharge_cur}C-RMSE:{rmse_value:.4f} V")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            fig.savefig(f"./simu_fig/{subdir_name}/{file_name_base}_{discharge_cur}C.png")
            plt.close(fig)

    return all_time_rmse


def run_with_timeout(param, soc_range, timeout=15):
    """
    Run simulation with timeout to avoid hanging processes.

    Args:
        param (np.ndarray): Battery parameter array
        soc_range (str or tuple): SOC range for RMSE calculation
        timeout (int): Timeout in seconds

    Returns:
        tuple: RMSE values, error reason
    """
    # Create a manager for sharing data between processes
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    # Define a function to catch errors during simulation
    def catch_error_simulation(solution, soc_range, return_dict):
        try:
            all_time_rmse = main_simulation(solution, soc_range)
            return_dict['result'] = all_time_rmse
            return_dict['reason'] = 'No Problem!'
        except Exception as e:
            print(f"Error occurred: {e}")
            return_dict['result'] = [1.5] * len(name.split(","))
            return_dict['reason'] = str(e)

    # Run the simulation in a separate process with timeout
    process = multiprocessing.Process(target=catch_error_simulation, args=(param, soc_range, return_dict))
    process.start()
    process.join(timeout)  # Wait for process to complete, with timeout

    if process.is_alive():
        process.terminate()  # Kill the process if it's still running
        process.join()  # Wait for process to truly end
        reason = "Timeout!"
        return [1.5] * len(name.split(",")), reason

    # Check for NaN in results
    result = return_dict.get('result')
    try:
        if any(np.isnan(result)):
            reason = "NaN values detected!"
            return [1.5] * len(name.split(",")), reason
        else:
            dict_reason = return_dict.get('reason')
            return result, dict_reason
    except Exception as e:
        print(f"Error checking results: {e}")
        reason = str(e)
        return [1.5] * len(name.split(",")), reason


def evaluate_objective(param):
    """
    Objective function for optimization - overall RMSE across all test conditions.

    Args:
        param (np.ndarray): Battery parameter array

    Returns:
        float: Maximum RMSE value across all test conditions
    """
    soc_range = 'all'
    all_time_rmse, reason = run_with_timeout(param, soc_range)
    obj = max(all_time_rmse)

    print(f"Normalized parameters: {param}")
    print(f"RMSE values (V): {all_time_rmse}")
    print(f"Objective value (max RMSE): {obj}")
    print(f"Status: {reason}")

    return obj


def optimize_without_constraints(initial_params):
    """
    Optimize battery parameters using CYIPOPT without constraints.

    Args:
        initial_params (np.ndarray): Initial parameter values from Bayesian optimization

    Returns:
        tuple: Optimized parameters, final objective value
    """
    print("Starting local optimization with IPOPT (unconstrained)...")
    print(f"Initial parameters: {initial_params}")

    class UnconstrainedBatteryProblem:
        """
        Class defining the battery parameter optimization problem for IPOPT without constraints.
        """

        def __init__(self, initial_params):
            """
            Initialize the optimization problem.

            Args:
                initial_params (np.ndarray): Initial parameter values from Bayesian optimization
            """
            self.initial_params = initial_params
            self.dim = len(initial_params)
            self.eval_count = 0

            # Record best solution
            self.best_params = None
            self.best_obj = float('inf')

            # Record all evaluations
            self.all_evaluations = []

        def objective(self, x):
            """
            Objective function for IPOPT.

            Args:
                x (np.ndarray): Battery parameter array

            Returns:
                float: Objective value (maximum RMSE)
            """
            self.eval_count += 1
            print(f"\nEvaluation #{self.eval_count}")

            obj = evaluate_objective(x)

            # Record evaluation
            self.all_evaluations.append({
                'params': x.copy(),
                'obj': obj
            })

            # Update best solution if needed
            if obj < self.best_obj:
                self.best_obj = obj
                self.best_params = x.copy()
                print(f"New best solution found! RMSE: {obj}")

            return obj

        def gradient(self, x):
            """
            Gradient approximation using finite differences.

            Args:
                x (np.ndarray): Battery parameter array

            Returns:
                np.ndarray: Gradient vector
            """
            # Use forward finite differences for efficiency
            epsilon = 1e-5
            grad = np.zeros(self.dim)

            # Base objective value
            f0 = self.objective(x)

            for i in range(self.dim):
                # Create perturbation vector
                h = np.zeros(self.dim)
                h[i] = epsilon

                # Forward evaluation (clipping to ensure bounds)
                x_plus = np.clip(x + h, 0, 1)
                f_plus = self.objective(x_plus)

                # Forward difference
                dx = x_plus[i] - x[i]  # Actual step size after clipping
                if abs(dx) > 1e-10:  # Protect against division by zero
                    grad[i] = (f_plus - f0) / dx
                else:
                    grad[i] = 0.0

            return grad

        def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
                         regularization_size, alpha_du, alpha_pr, ls_trials):
            """
            Callback function for IPOPT intermediate iterations.

            Args:
                Various IPOPT iteration data
            """
            print(f"Iteration {iter_count}: Objective = {obj_value}")
            print(f"Current best RMSE = {self.best_obj}")

            # Save intermediate results periodically
            if iter_count % 5 == 0:
                if self.best_params is not None:
                    save_results(self.best_params, self.best_obj,
                                 f"./solutions/Local/{file_name}_intermediate.json")

    # Create problem instance
    problem = UnconstrainedBatteryProblem(initial_params)

    # Define bounds (all parameters between 0 and 1)
    lb = np.zeros(problem.dim)
    ub = np.ones(problem.dim)

    # Create IPOPT problem (unconstrained)
    nlp = cyipopt.Problem(
        n=problem.dim,  # Number of variables
        m=0,  # Number of constraints (0 for unconstrained)
        problem_obj=problem,  # Problem instance
        lb=lb,  # Lower bounds on variables
        ub=ub  # Upper bounds on variables
    )

    # Set IPOPT options
    nlp.add_option('tol', 1e-4)
    nlp.add_option('max_iter', 50)
    nlp.add_option('print_level', 5)
    nlp.add_option('mu_strategy', 'adaptive')
    nlp.add_option('hessian_approximation', 'limited-memory')  # Use L-BFGS approximation

    # Run optimization
    try:
        x_opt, info = nlp.solve(initial_params)
        print("\nOptimization Results:")
        print(f"Final optimal parameters: {x_opt}")
        print(f"Final RMSE: {problem.best_obj}")
        print(f"Number of iterations: {info['iter_count']}")

        # Return the best parameters found during optimization
        return problem.best_params if problem.best_params is not None else x_opt, problem.best_obj
    except Exception as e:
        print(f"Optimization failed: {e}")

        if problem.best_params is not None:
            print("Returning best parameters found during optimization...")
            return problem.best_params, problem.best_obj
        else:
            print("Returning initial parameters...")
            return initial_params, evaluate_objective(initial_params)


def save_results(params, rmse, filename):
    """
    Save optimization results to a JSON file.

    Args:
        params (np.ndarray): Optimized battery parameters
        rmse (float): Final RMSE value
        filename (str): Output filename
    """
    # Ensure directory exists
    os.makedirs("./solutions/Local", exist_ok=True)

    result = {
        "parameters": params.tolist(),
        "rmse": rmse,
        "model_type": model_type,
        "battery_data": name
    }

    with open(filename, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"Results saved to {filename}")


def plot_comparison(bayes_params, ipopt_params):
    """
    Plot comparison between Bayesian optimization and IPOPT results.

    Args:
        bayes_params (np.ndarray): Parameters from Bayesian optimization
        ipopt_params (np.ndarray): Parameters after IPOPT optimization
    """
    # Run simulations with both parameter sets
    bayes_rmse = main_simulation(bayes_params, 'all', save=True, plot=True)
    ipopt_rmse = main_simulation(ipopt_params, 'all', save=True, plot=True)

    # Plot RMSE comparison
    names = name.split(",")
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, bayes_rmse, width, label='Bayesian Optimization')
    rects2 = ax.bar(x + width / 2, ipopt_rmse, width, label='IPOPT Optimization')

    param_names = [
        "Electrode height", "Electrode width", "Neg. electrode conductivity",
        "Pos. electrode diffusivity", "Pos. particle radius", "Initial conc. pos.",
        "Initial conc. neg.", "Pos. electrode conductivity", "Neg. particle radius",
        "Neg. electrode thickness", "Heat transfer coef.", "Separator density",
        "Separator thermal cond.", "Pos. electrode porosity", "Separator heat capacity",
        "Max conc. pos.", "Neg. electrode Bruggeman", "Pos. electrode Bruggeman",
        "Separator porosity", "Neg. collector thickness", "Pos. collector thickness",
        "Pos. electrode thickness", "Pos. active material fraction", "Neg. heat capacity",
        "Pos. thermal cond.", "Neg. active material fraction"
    ]

    ax.set_ylabel('RMSE (V)')
    ax.set_title('RMSE Comparison: Bayesian vs IPOPT Optimization')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"./solutions/Local/{file_name}_parameters.png", dpi=300)
    plt.close()


def main():
    """
    Main function to run local optimization starting from Bayesian results.
    """
    parser = argparse.ArgumentParser(description="Run local optimization using IPOPT")
    parser.add_argument('--filename', type=str, default="81#-T25-0.1C,81#-T25-0.2C,81#-T25-0.33C,81#-T25-1C",
                        help='Battery data files (comma-separated)')
    parser.add_argument('--model', type=str, choices=["DFN", "SPM"], default="DFN",
                        help='Battery model type')
    parser.add_argument('--bayes_result', type=str,
                        help='Path to Bayesian optimization results CSV')
    args = parser.parse_args()

    global name, model_type, file_name, subdir_name
    name = args.filename
    model_type = args.model
    file_name = name.split(",")[0].split("-")[0] + "MO-Local" + f"-{model_type}"
    subdir_name = "Local"

    # Create necessary directories
    os.makedirs(f"./solutions/{subdir_name}", exist_ok=True)
    os.makedirs(f"./simu_data/{subdir_name}", exist_ok=True)
    os.makedirs(f"./simu_fig/{subdir_name}", exist_ok=True)

    # Load initial solution from Bayesian optimization results
    bayes_csv = args.bayes_result
    if bayes_csv is None:
        # If not provided, try to construct the default path
        base_name = name.split(",")[0].split("-")[0]
        bayes_csv = f"./solutions/Bayes/{base_name}MO-Constraint-{model_type}.csv"

    print(f"Loading Bayesian optimization results from: {bayes_csv}")
    bayes_solution = read_csv_solution(bayes_csv)

    # Run unconstrained local optimization with IPOPT
    ipopt_solution, final_rmse = optimize_without_constraints(bayes_solution)

    # Save results
    save_results(ipopt_solution, final_rmse, f"./solutions/Local/{file_name}.json")

    # Plot comparison between Bayesian and IPOPT results
    plot_comparison(bayes_solution, ipopt_solution)

    print("\nOptimization completed successfully!")
    print(f"Final RMSE: {final_rmse}")


if __name__ == "__main__":
    main()
