# parameter_sensitivity.py
import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from matplotlib.colors import Normalize
from simulate_discharge import read_file_safe, pybamm_sim_fixed_params

# Parameter ranges based on the provided table
PARAMETER_RANGES = {
    "N_parallel": [180, 220],
    "electrode_height": [0.17, 0.22],
    "electrode_width": [0.15, 0.19],
    "Negative_electrode_thickness": [8e-5, 1.2e-4],
    "Positive_electrode_thickness": [9e-5, 1.3e-4],
    "Negative_particle_radius": [2e-6, 5e-6],
    "Positive_particle_radius": [1e-6, 4e-6],
    "Negative_electrode_active_material_volume_fraction": [0.48, 0.62],
    "Positive_electrode_active_material_volume_fraction": [0.45, 0.6],
    "Negative_electrode_porosity": [0.32, 0.45],
    "Positive_electrode_porosity": [0.32, 0.45],
    "Separator_porosity": [0.4, 0.6],
    "Maximum_concentration_in_negative_electrode": [25000, 35000],
    "Maximum_concentration_in_positive_electrode": [45000, 58000],
    "Negative_electrode_diffusivity": [1e-13, 1e-12],
    "Positive_electrode_diffusivity": [1e-13, 1e-12],
    "Negative_electrode_Bruggeman_coefficient": [1.2, 2.0],
    "Positive_electrode_Bruggeman_coefficient": [1.2, 2.0],
    "Negative_electrode_conductivity": [50, 150],
    "Positive_electrode_conductivity": [30, 80],
    "Initial_concentration_in_negative_electrode": [4000, 6000],
    "Initial_concentration_in_positive_electrode": [25000, 32000],
}

# Base parameter set (using the denormalized optimal parameters)
BASE_PARAMETERS = {
    "N_parallel": 191.63,
    "electrode_height": 0.2029,
    "electrode_width": 0.1582,
    "Negative_electrode_thickness": 1.0981e-4,
    "Positive_electrode_thickness": 1.0493e-4,
    "Negative_current_collector_thickness": 15e-6,  # fixed
    "Positive_current_collector_thickness": 20e-6,  # fixed
    "Negative_particle_radius": 3.9124e-6,
    "Positive_particle_radius": 3.0645e-6,
    "Negative_electrode_active_material_volume_fraction": 0.5642,
    "Positive_electrode_active_material_volume_fraction": 0.5236,
    "Negative_electrode_porosity": 0.4058,
    "Positive_electrode_porosity": 0.3542,
    "Separator_porosity": 0.5198,
    "Maximum_concentration_in_negative_electrode": 32750.0,
    "Maximum_concentration_in_positive_electrode": 51375.0,
    "Negative_electrode_diffusivity": 9.0872e-13,
    "Positive_electrode_diffusivity": 1.4804e-13,
    "Negative_electrode_Bruggeman_coefficient": 1.9318,
    "Positive_electrode_Bruggeman_coefficient": 1.6100,
    "Negative_electrode_conductivity": 79.63,
    "Positive_electrode_conductivity": 50.0,
    "Initial_concentration_in_negative_electrode": 4607.0,
    "Initial_concentration_in_positive_electrode": 28750.0,
}


def generate_parameter_values(param_name, num_points=40):
    """Generate evenly spaced parameter values within the defined range."""
    if param_name not in PARAMETER_RANGES:
        raise ValueError(f"Parameter {param_name} not found in defined ranges.")

    param_min, param_max = PARAMETER_RANGES[param_name]
    return np.linspace(param_min, param_max, num_points)


def worker_function(param_value, param_name, c_rates, base_params, battery_id, temperature, model_type, data_dir):
    """Worker function to run simulation for a specific parameter value across all C-rates."""
    results = {}

    # Create modified parameters for this simulation
    modified_params = base_params.copy()
    modified_params[param_name] = param_value

    for c_rate in c_rates:
        # Read experimental file details
        exp_file_name = os.path.join(data_dir, f"{battery_id}-T{temperature}-{c_rate}C.csv")
        time_max, max_voltage, min_voltage, capacity, _, _ = read_file_safe(exp_file_name)

        # Run simulation
        solution = pybamm_sim_fixed_params(
            fixed_params=modified_params,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            discharge_cur=c_rate,
            time_max=time_max,
            capacity=capacity,
            temperature=temperature,
            model_type=model_type
        )

        if solution:
            # Extract simulation results
            time_sim = solution["Time [s]"].entries
            voltage_sim = solution["Voltage [V]"].entries

            results[c_rate] = {
                'time': time_sim.tolist(),  # Convert to list for JSON serialization
                'voltage': voltage_sim.tolist()
            }
        else:
            results[c_rate] = None
            print(f"Simulation failed for {param_name}={param_value} at {c_rate}C")

    return param_value, results


def run_parameter_sweep(param_name, battery_id="81#", temperature=25, model_type="DFN",
                        data_dir="./bat_data", num_points=40, num_processes=None):
    """Run simulations with varying values for the specified parameter using multiprocessing."""
    # Create output directories
    output_dir = f"./sensitivity_results/plot_change/{param_name}"
    os.makedirs(output_dir, exist_ok=True)

    # C-rates to simulate
    c_rates = [0.1, 0.2, 0.33, 1]

    # Generate parameter values
    param_values = generate_parameter_values(param_name, num_points)

    # Set up number of processes (default to CPU count if not specified)
    if num_processes is None:
        num_processes = mp.cpu_count()

    # Create a pool of workers
    print(f"Running parameter sweep using {num_processes} processes")

    # Set up the worker function with fixed arguments
    worker_func = partial(
        worker_function,
        param_name=param_name,
        c_rates=c_rates,
        base_params=BASE_PARAMETERS,
        battery_id=battery_id,
        temperature=temperature,
        model_type=model_type,
        data_dir=data_dir
    )

    # Use a multiprocessing pool to run the simulations in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Map the worker function to each parameter value with progress bar
        results_list = list(tqdm(
            pool.imap(worker_func, param_values),
            total=len(param_values),
            desc=f"Simulating {param_name} sweep"
        ))

    # Process results
    processed_results = {c_rate: {} for c_rate in c_rates}

    for param_value, result in results_list:
        for c_rate, data in result.items():
            if data is not None:
                # Store the raw time and voltage data under this parameter value
                processed_results[c_rate][param_value] = {
                    'time': data['time'],
                    'voltage': data['voltage']
                }

    # Save results as single CSV files for each C-rate with parameter values as columns
    for c_rate in c_rates:
        # Create a DataFrame with time as the index
        # First, find the maximum length of time for any parameter value
        max_time_length = 0
        all_time_arrays = []

        for param_value, data in processed_results[c_rate].items():
            time_array = np.array(data['time'])
            all_time_arrays.append(time_array)
            max_time_length = max(max_time_length, len(time_array))

        # Create a common time array by interpolating to the maximum length
        if all_time_arrays:
            # Find the maximum simulation time across all results
            max_sim_time = max([time_array[-1] if len(time_array) > 0 else 0
                                for time_array in all_time_arrays])

            # Create a common time array from 0 to max_sim_time
            common_time = np.linspace(0, max_sim_time, max_time_length)

            # Create a DataFrame with time as the first column
            df = pd.DataFrame({'time': common_time})

            # Add voltage data for each parameter value as a column
            for param_value, data in processed_results[c_rate].items():
                # Get original time and voltage arrays
                orig_time = np.array(data['time'])
                orig_voltage = np.array(data['voltage'])

                if len(orig_time) > 1:  # Make sure we have at least 2 points for interpolation
                    # Interpolate voltage to the common time grid
                    # Use np.interp to handle the case where simulation ends early
                    interp_voltage = np.interp(
                        common_time,
                        orig_time,
                        orig_voltage,
                        right=np.nan  # Set values beyond the end of simulation to NaN
                    )

                    # Add to DataFrame with parameter value as column name
                    df[f"{param_value:.6g}"] = interp_voltage

            # Save to CSV
            csv_file = os.path.join(output_dir, f"{param_name}_{c_rate}C.csv")
            df.to_csv(csv_file, index=False)
            print(f"Saved data to {csv_file}")

    # Also save parameter values list
    param_values_file = os.path.join(output_dir, f"{param_name}_param_values.csv")
    pd.DataFrame({'param_values': param_values}).to_csv(param_values_file, index=False)

    print(f"Saved parameter sweep data to {output_dir}")

    # Create and save plots
    plot_parameter_sweep(param_name, processed_results, param_values, c_rates, output_dir)

    return processed_results


def plot_parameter_sweep(param_name, results, param_values, c_rates, output_dir):
    """Create plots showing the effect of parameter variation on discharge curves."""
    # Use plt.colormaps instead of get_cmap
    cmap = plt.colormaps['viridis']

    # Reduce the number of lines to plot for clarity (show 10 curves)
    plot_indices = np.linspace(0, len(param_values) - 1, 10, dtype=int)
    norm = Normalize(vmin=min(param_values), vmax=max(param_values))

    # Create plot for each C-rate
    for c_rate in c_rates:
        c_rate_results = results[c_rate]
        if not c_rate_results:
            continue

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot each selected parameter value
        for i, idx in enumerate(plot_indices):
            param_value = param_values[idx]

            if param_value in c_rate_results:
                # Get the time and voltage data
                time_data = c_rate_results[param_value]['time']
                voltage_data = c_rate_results[param_value]['voltage']

                # Plot the discharge curve with color corresponding to parameter value
                color = cmap(i / len(plot_indices))
                ax.plot(time_data, voltage_data, color=color,
                        label=f"{param_name}={param_value:.4g}")

        # Format the plot
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Terminal Voltage [V]')
        ax.set_title(f'Discharge Curves at {c_rate}C: Effect of {param_name}')
        ax.grid(True)

        # Add colorbar to indicate parameter values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label=param_name)

        # Save the plot
        plot_file = os.path.join(output_dir, f"{param_name}_{c_rate}C.png")
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300)
        plt.close(fig)

        print(f"Saved plot to {plot_file}")

    # Create a combined plot with all C-rates
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()

    for i, c_rate in enumerate(c_rates):
        if i >= len(axs) or not c_rate in results:
            continue

        c_rate_results = results[c_rate]
        if not c_rate_results:
            continue

        ax = axs[i]

        # Plot each selected parameter value
        for j, idx in enumerate(plot_indices):
            param_value = param_values[idx]

            if param_value in c_rate_results:
                time_data = c_rate_results[param_value]['time']
                voltage_data = c_rate_results[param_value]['voltage']

                color = cmap(j / len(plot_indices))
                ax.plot(time_data, voltage_data, color=color)

        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Terminal Voltage [V]')
        ax.set_title(f'{c_rate}C Discharge')
        ax.grid(True)

    # Add a common colorbar
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label=param_name)

    plt.suptitle(f'Effect of {param_name} on Discharge Curves at Different C-rates', fontsize=16)

    # Save the combined plot
    plot_file = os.path.join(output_dir, f"{param_name}_all_C_rates.png")
    plt.savefig(plot_file, dpi=300)
    plt.close(fig)

    print(f"Saved combined plot to {plot_file}")


def main():
    parser = argparse.ArgumentParser(description="Run parameter sensitivity analysis for PyBaMM battery simulations.")
    parser.add_argument('--param', type=str, default='N_parallel', help='Parameter name to vary')
    parser.add_argument('--battery_id', type=str, default="81#", help='Identifier for the battery (e.g., 81#).')
    parser.add_argument('--temperature', type=int, default=25, help='Operating temperature in Celsius.')
    parser.add_argument('--model', type=str, choices=["DFN", "SPM"], default="DFN", help='PyBaMM Model Type.')
    parser.add_argument('--data_dir', type=str, default="./bat_data", help='Directory with experimental CSV files.')
    parser.add_argument('--num_points', type=int, default=40, help='Number of parameter values to simulate.')
    parser.add_argument('--num_processes', type=int, default=None, help='Number of processes to use (default: CPU count).')

    args = parser.parse_args()
    param_lsit = ['N_parallel', 'Negative_electrode_thickness', 'Negative_electrode_active_material_volume_fraction',
                  'electrode_height', 'Initial_concentration_in_positive_electrode', 'Negative_electrode_conductivity']
    for param in param_lsit:
        args.param = param
        # Validate parameter name
        if args.param not in PARAMETER_RANGES:
            print(f"Error: Parameter '{args.param}' not found. Available parameters:")
            for param in PARAMETER_RANGES.keys():
                print(f"  - {param}")
            return

        # Run parameter sweep with multiprocessing
        run_parameter_sweep(
            param_name=args.param,
            battery_id=args.battery_id,
            temperature=args.temperature,
            model_type=args.model,
            data_dir=args.data_dir,
            num_points=args.num_points,
            num_processes=args.num_processes
        )


if __name__ == '__main__':
    # This is important for Windows to avoid recursive creation of processes
    mp.freeze_support()
    main()
