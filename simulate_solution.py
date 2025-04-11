import pybamm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os

# Constants for default values if experimental file reading fails
DEFAULT_CAPACITY = 280.0  # Ah
DEFAULT_MAX_VOLTAGE = 3.5  # V
DEFAULT_MIN_VOLTAGE = 2.5  # V
DEFAULT_TIME_MAX_ESTIMATE_FACTOR = 1.2  # Estimate max time = (1/C_rate) * 3600 * factor


def min_max_func(min_val, max_val, norm_val):
    """Convert normalized value (0-1) to actual value in the given range."""
    return min_val + norm_val * (max_val - min_val)


def read_csv_solution(csv_file, i):
    """Read a specific solution (row) from a CSV file."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file)

        # Get the specified row solution
        solution_str = df.iloc[i]['Solution']
        cleaned_str = solution_str.strip('[]')  # Remove possible brackets and spaces
        numbers = cleaned_str.split()  # Split by space
        solution = np.array(list(map(float, numbers)))  # Convert to array of floats

        print(f"Successfully loaded solution from {csv_file}")
        print(f"Solution shape: {solution.shape}")
        print(f"Solution: {solution}")

        return solution
    except Exception as e:
        print(f"Error reading CSV solution: {e}")
        raise


def read_file_safe(file_name):
    """Safely reads experimental file details, providing defaults if file not found."""
    try:
        data = pd.read_csv(file_name)
        # Extract time and voltage data for plotting
        time_data = data['time'].values if not data.empty else np.array([])
        voltage_data = data['V'].values if not data.empty else np.array([])

        # Estimate time_max more robustly, handle empty files
        time_max = data['time'].values[-1] if not data.empty else None
        voltage_max = data['V'].values[0] if not data.empty else DEFAULT_MAX_VOLTAGE
        # Ensure voltage_min is calculated reasonably even if data is short
        voltage_min = data['V'].values[-1] if not data.empty else DEFAULT_MIN_VOLTAGE
        capacity = data['Ah'].values[-1] if not data.empty else DEFAULT_CAPACITY

        # If time_max couldn't be determined, estimate based on C-rate in filename
        if time_max is None:
            try:
                discharge_cur = float(file_name.split('-')[-1].replace('C.csv', ''))
                time_max = (1.0 / discharge_cur) * 3600 * DEFAULT_TIME_MAX_ESTIMATE_FACTOR  # Estimate in seconds
            except:
                time_max = 3600 * 10  # Default fallback (e.g., 10 hours for 0.1C)
            print(f"Warning: Could not read time_max from {file_name}. Estimated as {time_max:.0f}s.")

        print(f"Read parameters from {file_name}: Capacity={capacity:.2f}Ah, V_max={voltage_max:.2f}V, V_min={voltage_min:.2f}V, Time_max={time_max:.0f}s")
        return time_max, voltage_max, voltage_min, capacity, time_data, voltage_data
    except FileNotFoundError:
        print(f"Warning: File not found: {file_name}. Using default values.")
        try:
            # Attempt to get C-rate from filename for time estimation
            discharge_cur = float(file_name.split('-')[-1].replace('C.csv', ''))
            time_max_est = (1.0 / discharge_cur) * 3600 * DEFAULT_TIME_MAX_ESTIMATE_FACTOR
        except:
            time_max_est = 3600 * 10  # Default fallback
        print(f"Using defaults: Capacity={DEFAULT_CAPACITY}Ah, V_max={DEFAULT_MAX_VOLTAGE}V, V_min={DEFAULT_MIN_VOLTAGE}V, Estimated Time_max={time_max_est:.0f}s")
        return time_max_est, DEFAULT_MAX_VOLTAGE, DEFAULT_MIN_VOLTAGE, DEFAULT_CAPACITY, np.array([]), np.array([])
    except Exception as e:
        print(f"Error reading file {file_name}: {e}. Using default values.")
        try:
            discharge_cur = float(file_name.split('-')[-1].replace('C.csv', ''))
            time_max_est = (1.0 / discharge_cur) * 3600 * DEFAULT_TIME_MAX_ESTIMATE_FACTOR
        except:
            time_max_est = 3600 * 10
        print(f"Using defaults: Capacity={DEFAULT_CAPACITY}Ah, V_max={DEFAULT_MAX_VOLTAGE}V, V_min={DEFAULT_MIN_VOLTAGE}V, Estimated Time_max={time_max_est:.0f}s")
        return time_max_est, DEFAULT_MAX_VOLTAGE, DEFAULT_MIN_VOLTAGE, DEFAULT_CAPACITY, np.array([]), np.array([])


def convert_normalized_params_to_real(param):
    """Convert normalized parameters (0-1) to their real physical values."""
    # Geometry and structure parameters
    N_parallel = min_max_func(180, 220, param[0])
    electrode_height = min_max_func(0.17, 0.22, param[1])
    electrode_width = min_max_func(0.15, 0.19, param[2])
    Negative_electrode_thickness = min_max_func(80e-6, 120e-6, param[3])
    Positive_electrode_thickness = min_max_func(90e-6, 130e-6, param[4])

    # Material composition parameters
    Positive_electrode_active_material_volume_fraction = min_max_func(0.45, 0.6, param[5])
    Negative_electrode_active_material_volume_fraction = min_max_func(0.48, 0.62, param[6])
    Positive_electrode_porosity = min_max_func(0.32, 0.45, param[7])
    Negative_electrode_porosity = min_max_func(0.32, 0.45, param[8])
    Separator_porosity = min_max_func(0.4, 0.6, param[9])

    # Transport properties
    Positive_electrode_diffusivity = min_max_func(1e-13, 1e-12, param[10])
    Negative_electrode_diffusivity = min_max_func(1e-13, 1e-12, param[11])
    Positive_particle_radius = min_max_func(1e-6, 4e-6, param[12])
    Negative_particle_radius = min_max_func(2e-6, 5e-6, param[13])
    Negative_electrode_conductivity = min_max_func(50.0, 150.0, param[14])
    Positive_electrode_conductivity = min_max_func(30.0, 80.0, param[15])
    Negative_electrode_Bruggeman_coefficient = min_max_func(1.2, 2.0, param[16])
    Positive_electrode_Bruggeman_coefficient = min_max_func(1.2, 2.0, param[17])

    # Concentration parameters
    Initial_concentration_in_positive_electrode = min_max_func(25000.0, 32000.0, param[18])
    Initial_concentration_in_negative_electrode = min_max_func(4000.0, 6000.0, param[19])
    Maximum_concentration_in_positive_electrode = min_max_func(45000.0, 58000.0, param[20])
    Maximum_concentration_in_negative_electrode = min_max_func(25000.0, 35000.0, param[21])

    # Print the denormalized parameters
    print("\nDenormalized Parameters:")
    print(f"N_parallel: {N_parallel:.2f}")
    print(f"electrode_height: {electrode_height:.4f} m")
    print(f"electrode_width: {electrode_width:.4f} m")
    print(f"Negative_electrode_thickness: {Negative_electrode_thickness * 1e6:.2f} μm")
    print(f"Positive_electrode_thickness: {Positive_electrode_thickness * 1e6:.2f} μm")
    print(f"Positive_electrode_active_material_volume_fraction: {Positive_electrode_active_material_volume_fraction:.4f}")
    print(f"Negative_electrode_active_material_volume_fraction: {Negative_electrode_active_material_volume_fraction:.4f}")
    print(f"Positive_electrode_porosity: {Positive_electrode_porosity:.4f}")
    print(f"Negative_electrode_porosity: {Negative_electrode_porosity:.4f}")
    print(f"Separator_porosity: {Separator_porosity:.4f}")
    print(f"Positive_electrode_diffusivity: {Positive_electrode_diffusivity:.2e} m²/s")
    print(f"Negative_electrode_diffusivity: {Negative_electrode_diffusivity:.2e} m²/s")
    print(f"Positive_particle_radius: {Positive_particle_radius * 1e6:.2f} μm")
    print(f"Negative_particle_radius: {Negative_particle_radius * 1e6:.2f} μm")
    print(f"Negative_electrode_conductivity: {Negative_electrode_conductivity:.2f} S/m")
    print(f"Positive_electrode_conductivity: {Positive_electrode_conductivity:.2f} S/m")
    print(f"Negative_electrode_Bruggeman_coefficient: {Negative_electrode_Bruggeman_coefficient:.2f}")
    print(f"Positive_electrode_Bruggeman_coefficient: {Positive_electrode_Bruggeman_coefficient:.2f}")
    print(f"Initial_concentration_in_positive_electrode: {Initial_concentration_in_positive_electrode:.2f} mol/m³")
    print(f"Initial_concentration_in_negative_electrode: {Initial_concentration_in_negative_electrode:.2f} mol/m³")
    print(f"Maximum_concentration_in_positive_electrode: {Maximum_concentration_in_positive_electrode:.2f} mol/m³")
    print(f"Maximum_concentration_in_negative_electrode: {Maximum_concentration_in_negative_electrode:.2f} mol/m³")

    # Create parameter dictionary
    param_dict = {
        "Number of electrodes connected in parallel to make a cell": N_parallel,
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

        # Transport properties
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

    return param_dict


def plot_simulation_result(time_simulation, voltage_simulation, time_experimental, voltage_experimental,
                           discharge_cur, temperature, battery_id, model_type, output_dir, solution_index):
    """Plots the simulation results and experimental data if available."""
    fig, ax = plt.subplots()

    # Plot simulation data
    ax.plot(time_simulation, voltage_simulation, linestyle='-', label=f'{discharge_cur}C Simulation', color='blue')

    # Plot experimental data if available
    if len(time_experimental) > 0 and len(voltage_experimental) > 0:
        ax.plot(time_experimental, voltage_experimental, linestyle='--', label=f'{discharge_cur}C Experimental', color='red')

    plt.xlabel('Time [s]')
    plt.ylabel('Terminal Voltage [V]')
    plt.title(f"{battery_id}-T{temperature}-{discharge_cur}C ({model_type}) - Solution {solution_index}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Ensure output directory exists
    fig_dir = os.path.join(output_dir, "simu_fig")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"{battery_id}-T{temperature}-{discharge_cur}C-{model_type}-Sol{solution_index}.png")
    fig.savefig(fig_path)
    print(f"Saved plot to: {fig_path}")
    plt.close(fig)  # Close the figure to free memory


def pybamm_sim_with_params(parameter_dict, min_voltage, max_voltage, discharge_cur, time_max, capacity, temperature, model_type):
    """Runs a PyBaMM simulation with the provided parameter dictionary."""
    parameter_values = pybamm.ParameterValues("Prada2013")  # Use appropriate base parameter set

    # Add voltage limits and temperature settings
    parameter_dict.update({
        "Nominal cell capacity [A.h]": capacity,
        "Lower voltage cut-off [V]": min_voltage - 0.1,  # Give some buffer
        "Upper voltage cut-off [V]": max_voltage + 0.1,  # Give some buffer
        "Ambient temperature [K]": 273.15 + 25,  # Assuming ambient is 25C
        "Initial temperature [K]": 273.15 + temperature,
    })

    # --- Select Model ---
    option = {"cell geometry": "arbitrary", "thermal": "lumped"}  # Keep options simple
    if model_type == "DFN":
        model = pybamm.lithium_ion.DFN()
    elif model_type == "SPM":
        model = pybamm.lithium_ion.SPM()
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'DFN' or 'SPM'.")

    # --- Define Experiment ---
    exp = pybamm.Experiment(
        [(f"Discharge at {discharge_cur} C for {time_max} seconds",)]
    )

    # Update parameter values
    parameter_values.update(parameter_dict, check_already_exists=False)

    # --- Setup and Run Simulation ---
    safe_solver = pybamm.CasadiSolver(mode="safe", dt_max=60)  # Adjust dt_max if needed
    sim = pybamm.Simulation(model, parameter_values=parameter_values, solver=safe_solver, experiment=exp)

    print(f"Starting simulation for {discharge_cur}C...")
    try:
        sol = sim.solve(initial_soc=1.0)  # Start from fully charged
        print(f"Simulation for {discharge_cur}C finished.")
        return sol
    except Exception as e:
        print(f"Error during simulation for {discharge_cur}C: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Run PyBaMM simulations with parameters from a CSV solution file.")
    parser.add_argument('--solution_file', type=str, default="solutions/Bayes/81#MO-Constraint-DFN-22.csv",
                        help='Path to the CSV file containing the solution parameters.')
    parser.add_argument('--solution_index', type=int, default=0,
                        help='Index of the solution row to use from the CSV file (default: 0 for first row).')
    parser.add_argument('--battery_id', type=str, default="81#", help='Identifier for the battery (e.g., 81#).')
    parser.add_argument('--temperature', type=int, default=25, help='Operating temperature in Celsius.')
    parser.add_argument('--model', type=str, choices=["DFN", "SPM"], default="DFN", help='PyBaMM Model Type (DFN or SPM).')
    parser.add_argument('--data_dir', type=str, default="./bat_data", help='Directory containing experimental CSV files.')
    parser.add_argument('--output_dir', type=str, default="./simulation_output", help='Directory to save simulation results.')
    parser.add_argument('--save_data', action='store_true', default=True, help='Save simulation time/voltage data to CSV.')

    args = parser.parse_args()

    # Read the solution from the CSV file
    print(f"Reading solution from {args.solution_file}, using row {args.solution_index}")
    normalized_params = read_csv_solution(args.solution_file, args.solution_index)

    # Convert normalized parameters to real physical values
    parameters_dict = convert_normalized_params_to_real(normalized_params)

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "simu_data"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "simu_fig"), exist_ok=True)

    # Define the C-rates to simulate
    c_rates = [0.1, 0.2, 0.33, 1.0]

    # Loop through each C-rate
    for c_rate in c_rates:
        print(f"\n--- Simulating {c_rate}C Discharge ---")
        # Construct the expected experimental filename to read capacity/limits
        exp_file_name = os.path.join(args.data_dir, f"{args.battery_id}-T{args.temperature}-{c_rate}C.csv")

        # Read experimental file details (or use defaults)
        time_max, max_voltage, min_voltage, capacity, time_exp, voltage_exp = read_file_safe(exp_file_name)

        # Run the simulation with the parameters
        solution = pybamm_sim_with_params(
            parameter_dict=parameters_dict,
            min_voltage=min_voltage,
            max_voltage=max_voltage,
            discharge_cur=c_rate,
            time_max=time_max,
            capacity=capacity,
            temperature=args.temperature,
            model_type=args.model
        )

        if solution:
            # Extract simulation results
            time_sim = solution["Time [s]"].entries
            voltage_sim = solution["Voltage [V]"].entries

            # Plot the results (both simulation and experimental if available)
            plot_simulation_result(
                time_sim, voltage_sim,
                time_exp, voltage_exp,
                c_rate, args.temperature, args.battery_id, args.model,
                args.output_dir, args.solution_index
            )

            # Save data if requested
            if args.save_data:
                data_dir = os.path.join(args.output_dir, "simu_data")
                data_path = os.path.join(data_dir, f"{args.battery_id}-T{args.temperature}-{c_rate}C-{args.model}-Sol{args.solution_index}.csv")

                # If we have experimental data, include it in the saved file
                if len(time_exp) > 0 and len(voltage_exp) > 0:
                    df_sim = pd.DataFrame({
                        "real_time": np.interp(np.linspace(0, 1, len(time_sim)),
                                               np.linspace(0, 1, len(time_exp)),
                                               time_exp) if len(time_exp) > 1 else np.nan,
                        "real_voltage": np.interp(np.linspace(0, 1, len(time_sim)),
                                                  np.linspace(0, 1, len(voltage_exp)),
                                                  voltage_exp) if len(voltage_exp) > 1 else np.nan,
                        "simu_time": time_sim,
                        "simu_voltage": voltage_sim
                    })
                else:
                    df_sim = pd.DataFrame({"Time [s]": time_sim, "Voltage [V]": voltage_sim})

                df_sim.to_csv(data_path, index=False)
                print(f"Saved simulation data to: {data_path}")
        else:
            print(f"Simulation failed for {c_rate}C.")

    print("\nAll simulations completed.")


if __name__ == '__main__':
    main()
