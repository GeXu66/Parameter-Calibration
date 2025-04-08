# sensitivity_analysis_mp.py
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import time
import multiprocessing as mp
from functools import partial
from SALib.sample import saltelli
from SALib.analyze import sobol
from tqdm import tqdm
import simulate_discharge as sim  # Import your existing simulation module


def define_problem():
    """Define the problem for SALib with parameter ranges from the provided table."""
    return {
        'num_vars': 22,  # Total number of parameters (excluding 2 fixed ones)
        'names': [
            # Geometric Parameters
            'N_parallel',
            'electrode_height',
            'electrode_width',
            'Negative_electrode_thickness',
            'Positive_electrode_thickness',

            # Structural Parameters
            'Negative_particle_radius',
            'Positive_particle_radius',
            'Negative_electrode_active_material_volume_fraction',
            'Positive_electrode_active_material_volume_fraction',
            'Negative_electrode_porosity',
            'Positive_electrode_porosity',
            'Separator_porosity',
            'Maximum_concentration_in_negative_electrode',
            'Maximum_concentration_in_positive_electrode',

            # Transport Parameters
            'Negative_electrode_diffusivity',
            'Positive_electrode_diffusivity',
            'Negative_electrode_Bruggeman_coefficient',
            'Positive_electrode_Bruggeman_coefficient',
            'Negative_electrode_conductivity',
            'Positive_electrode_conductivity',

            # Initial State Parameters
            'Initial_concentration_in_negative_electrode',
            'Initial_concentration_in_positive_electrode',
        ],
        'bounds': [
            # Geometric Parameters
            [150, 250],  # N_parallel
            [0.17, 0.22],  # electrode_height [m]
            [0.15, 0.19],  # electrode_width [m]
            [8e-5, 1.2e-4],  # Negative_electrode_thickness [m]
            [9e-5, 1.3e-4],  # Positive_electrode_thickness [m]

            # Structural Parameters
            [2e-6, 5e-6],  # Negative_particle_radius [m]
            [1e-6, 4e-6],  # Positive_particle_radius [m]
            [0.48, 0.62],  # Negative_electrode_active_material_volume_fraction
            [0.45, 0.6],  # Positive_electrode_active_material_volume_fraction
            [0.32, 0.45],  # Negative_electrode_porosity
            [0.32, 0.45],  # Positive_electrode_porosity
            [0.4, 0.6],  # Separator_porosity
            [25000, 35000],  # Maximum_concentration_in_negative_electrode [mol/m³]
            [45000, 58000],  # Maximum_concentration_in_positive_electrode [mol/m³]

            # Transport Parameters
            [1e-13, 1e-12],  # Negative_electrode_diffusivity [m²/s]
            [1e-13, 1e-12],  # Positive_electrode_diffusivity [m²/s]
            [1.2, 2.0],  # Negative_electrode_Bruggeman_coefficient
            [1.2, 2.0],  # Positive_electrode_Bruggeman_coefficient
            [50, 150],  # Negative_electrode_conductivity [S/m]
            [30, 80],  # Positive_electrode_conductivity [S/m]


            # Initial State Parameters
            [4000, 6000],  # Initial_concentration_in_negative_electrode [mol/m³]
            [25000, 32000],  # Initial_concentration_in_positive_electrode [mol/m³]
        ]
    }


def update_parameters(base_params, sample_values, problem):
    """Update the parameter dictionary with sample values for sensitivity analysis."""
    params = base_params.copy()

    # Map SALib sample values to parameter names
    for i, name in enumerate(problem['names']):
        # Handle special cases for parameter names that differ from PyBaMM names
        if name == 'Electrolyte_conductivity':
            params["Electrolyte conductivity [S.m-1]"] = sample_values[i]
        elif name == 'Electrolyte_diffusivity':
            params["Electrolyte diffusivity [m2.s-1]"] = sample_values[i]
        else:
            params[name] = sample_values[i]

    return params


def evaluate_model(params_sample, idx, problem, base_params, c_rate, battery_id, temperature, model_type, data_dir):
    """Worker function to evaluate the model with a single parameter set."""
    try:
        # Update parameters with sample values
        current_params = update_parameters(base_params, params_sample, problem)

        # Read experimental data
        exp_file_name = os.path.join(data_dir, f"{battery_id}-T{temperature}-{c_rate}C.csv")
        time_max, max_voltage, min_voltage, capacity, time_exp, voltage_exp = sim.read_file_safe(exp_file_name)

        # Skip evaluation if no experimental data available
        if len(time_exp) == 0 or len(voltage_exp) == 0:
            print(f"No experimental data for {c_rate}C, skipping evaluation for sample {idx}.")
            return idx, np.inf

        # Run simulation with provided parameters
        solution = sim.pybamm_sim_fixed_params(
            fixed_params=current_params,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            discharge_cur=c_rate,
            time_max=time_max,
            capacity=capacity,
            temperature=temperature,
            model_type=model_type
        )

        if solution is None:
            # Return a high error if simulation fails
            return idx, np.inf

        # Extract simulation results
        time_sim = solution["Time [s]"].entries
        voltage_sim = solution["Voltage [V]"].entries

        # Interpolate experimental data to match simulation time points for comparison
        if len(time_exp) > 1 and len(voltage_exp) > 1:
            voltage_exp_interp = np.interp(time_sim, time_exp, voltage_exp)

            # Calculate root mean square error between simulation and experiment
            rmse = np.sqrt(np.mean((voltage_sim - voltage_exp_interp) ** 2))
            return idx, rmse
        else:
            return idx, np.inf

    except Exception as e:
        print(f"Error in worker process (sample {idx}): {e}")
        return idx, np.inf


def run_sensitivity_analysis(args, base_params):
    """Run sensitivity analysis across multiple C-rates using multiprocessing."""
    # Define the problem
    problem = define_problem()

    # Generate samples using Saltelli method
    param_values = saltelli.sample(problem, args.n_samples)
    total_samples = param_values.shape[0]
    print(f"Generated {total_samples} parameter combinations for evaluation")

    # Define C-rates for analysis
    c_rates = [0.1, 0.2, 0.33, 1]

    # Get number of CPUs for parallel processing
    num_cpus = min(mp.cpu_count(), args.max_cpus)
    print(f"Using {num_cpus} CPU cores for parallel processing")

    # Create directory for results
    results_dir = "sensitivity_results"
    os.makedirs(results_dir, exist_ok=True)

    # Store all sensitivity results
    sensitivity_results = {}

    # Run analysis for each C-rate
    for c_rate in c_rates:
        print(f"\n--- Running sensitivity analysis for {c_rate}C ---")
        start_time = time.time()

        # Initialize array for storing results
        Y = np.zeros(total_samples)

        # Create a partial function with fixed parameters
        worker_func = partial(
            evaluate_model,
            problem=problem,
            base_params=base_params,
            c_rate=c_rate,
            battery_id=args.battery_id,
            temperature=args.temperature,
            model_type=args.model,
            data_dir=args.data_dir
        )

        # Create argument list for multiprocessing
        arg_list = [(param_values[i], i) for i in range(total_samples)]

        # Using multiprocessing to evaluate models in parallel
        print(f"Running {total_samples} model evaluations in parallel...")
        with mp.Pool(processes=num_cpus) as pool:
            # Use tqdm with imap to show progress
            results = list(tqdm(
                pool.starmap(worker_func, arg_list),
                total=total_samples,
                desc=f"C-rate {c_rate}"
            ))

        # Process results back into ordered array
        for idx, rmse in results:
            Y[idx] = rmse

        elapsed_time = time.time() - start_time
        print(f"Completed {c_rate}C evaluations in {elapsed_time:.1f} seconds")

        # Save raw model outputs
        pd.DataFrame({"sample_idx": range(len(Y)), "rmse": Y}).to_csv(
            os.path.join(results_dir, f"model_outputs_{c_rate}C.csv"), index=False
        )

        # Handle failed simulations by replacing infinite values
        Y[np.isinf(Y)] = np.max(Y[~np.isinf(Y)]) * 2 if np.any(~np.isinf(Y)) else 1000

        # Analyze with Sobol method
        Si = sobol.analyze(problem, Y, print_to_console=True)

        # Save sensitivity indices
        results = {
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf']
        }
        sensitivity_results[c_rate] = results

        # Save to CSV
        results_df = pd.DataFrame({
            'Parameter': problem['names'],
            'S1': Si['S1'],
            'S1_conf': Si['S1_conf'],
            'ST': Si['ST'],
            'ST_conf': Si['ST_conf']
        })
        results_df.to_csv(os.path.join(results_dir, f"sensitivity_indices_{c_rate}C.csv"), index=False)

        # Plot sensitivity indices
        plot_sensitivity_indices(Si, problem, c_rate, results_dir)

    # Compute weighted sensitivity indices across all C-rates
    # Use weights that emphasize higher C-rates where sensitivity is often more pronounced
    weights = {0.1: 0.1, 0.2: 0.2, 0.33: 0.3, 1: 0.4}

    # Initialize weighted results
    weighted_S1 = np.zeros(len(problem['names']))
    weighted_ST = np.zeros(len(problem['names']))

    # Calculate weighted average
    for c_rate, result in sensitivity_results.items():
        weighted_S1 += weights[c_rate] * result['S1']
        weighted_ST += weights[c_rate] * result['ST']

    # Create and save weighted results
    weighted_results = pd.DataFrame({
        'Parameter': problem['names'],
        'Weighted_S1': weighted_S1,
        'Weighted_ST': weighted_ST
    })
    weighted_results.to_csv(os.path.join(results_dir, "weighted_sensitivity_indices.csv"), index=False)

    # Plot weighted sensitivity indices
    plot_weighted_sensitivity(weighted_results, results_dir)

    print(f"\nSensitivity analysis complete. Results saved to {results_dir}")


def plot_sensitivity_indices(Si, problem, c_rate, results_dir):
    """Plot first-order and total-order sensitivity indices."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 12))  # Increased height for more parameters

    # Sort indices by value
    S1_indices = np.argsort(Si['S1'])
    ST_indices = np.argsort(Si['ST'])

    # First-order indices
    ax1.barh(np.arange(len(problem['names'])), Si['S1'][S1_indices], xerr=Si['S1_conf'][S1_indices])
    ax1.set_yticks(np.arange(len(problem['names'])))
    ax1.set_yticklabels([problem['names'][i] for i in S1_indices])
    ax1.set_title(f'First-Order Sensitivity Indices - {c_rate}C')
    ax1.set_xlabel('S1')

    # Total-order indices
    ax2.barh(np.arange(len(problem['names'])), Si['ST'][ST_indices], xerr=Si['ST_conf'][ST_indices])
    ax2.set_yticks(np.arange(len(problem['names'])))
    ax2.set_yticklabels([problem['names'][i] for i in ST_indices])
    ax2.set_title(f'Total-Order Sensitivity Indices - {c_rate}C')
    ax2.set_xlabel('ST')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f"sensitivity_plot_{c_rate}C.png"), dpi=300)
    plt.close()


def plot_weighted_sensitivity(weighted_results, results_dir):
    """Plot weighted sensitivity indices across all C-rates."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 12))  # Increased height for more parameters

    # Sort by weighted values
    sorted_S1 = weighted_results.sort_values('Weighted_S1')
    sorted_ST = weighted_results.sort_values('Weighted_ST')

    # First-order weighted indices
    ax1.barh(np.arange(len(sorted_S1)), sorted_S1['Weighted_S1'])
    ax1.set_yticks(np.arange(len(sorted_S1)))
    ax1.set_yticklabels(sorted_S1['Parameter'])
    ax1.set_title('Weighted First-Order Sensitivity Indices (Across All C-rates)')
    ax1.set_xlabel('Weighted S1')

    # Total-order weighted indices
    ax2.barh(np.arange(len(sorted_ST)), sorted_ST['Weighted_ST'])
    ax2.set_yticks(np.arange(len(sorted_ST)))
    ax2.set_yticklabels(sorted_ST['Parameter'])
    ax2.set_title('Weighted Total-Order Sensitivity Indices (Across All C-rates)')
    ax2.set_xlabel('Weighted ST')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "weighted_sensitivity_plot.png"), dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run sensitivity analysis for PyBaMM battery simulations.")
    parser.add_argument('--battery_id', type=str, default="81#", help='Identifier for the battery (e.g., 81#).')
    parser.add_argument('--temperature', type=int, default=25, help='Operating temperature in Celsius.')
    parser.add_argument('--model', type=str, choices=["DFN", "SPM"], default="DFN", help='PyBaMM Model Type (DFN or SPM).')
    parser.add_argument('--data_dir', type=str, default="./bat_data", help='Directory containing experimental CSV files.')
    parser.add_argument('--n_samples', type=int, default=128, help='Number of samples per parameter (Saltelli method). Default reduced to 8 due to high dimensionality.')
    parser.add_argument('--max_cpus', type=int, default=mp.cpu_count(), help=f'Maximum number of CPU cores to use. Default: all ({mp.cpu_count()} cores).')
    parser.add_argument('--skip_c_rates', type=str, default="", help='Comma-separated list of C-rates to skip (e.g., "0.1,0.2").')

    args = parser.parse_args()

    # Base parameters that won't be varied in sensitivity analysis
    base_params = {"Negative_current_collector_thickness": 1.5e-5, "Positive_current_collector_thickness": 2.0e-5,
                   "Separator_porosity": 0.5, "Separator_density": 1100.0, "Separator_specific_heat_capacity": 1900.0}

    # Add current collector thicknesses which are fixed per the table

    # Print start message with configuration details
    print(f"\n{'=' * 80}")
    print(f"BATTERY PARAMETER SENSITIVITY ANALYSIS WITH MULTIPROCESSING")
    print(f"{'=' * 80}")
    print(f"Configuration:")
    print(f"  - Battery ID: {args.battery_id}")
    print(f"  - Temperature: {args.temperature}°C")
    print(f"  - Model type: {args.model}")
    print(f"  - Sample size: {args.n_samples} (generating {(2 * args.n_samples + 2) * 34} total model evaluations per C-rate)")
    print(f"  - Using up to {args.max_cpus} CPU cores")
    print(f"  - Data directory: {args.data_dir}")
    print(f"{'=' * 80}\n")

    # Start processing time
    total_start_time = time.time()

    # Run sensitivity analysis
    run_sensitivity_analysis(args, base_params)

    # Report total time
    total_elapsed = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal analysis time: {int(hours)}h {int(minutes)}m {seconds:.1f}s")


if __name__ == '__main__':
    main()
