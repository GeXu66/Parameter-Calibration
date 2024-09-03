import pybamm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors, colormaps

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    # Load a lithium-ion model
    param_name = "Positive electrode active material volume fraction"

    # Define the range of values for the parameter
    min_param = 0.5
    max_param = 0.7
    param_values = np.linspace(min_param, max_param, 3)
    fig, ax = plt.subplots()
    # define color map
    cmap = colormaps["Spectral"]
    # need to normalize because color maps are defined in [0, 1]
    norm = colors.Normalize(min_param, max_param)
    param_list = ["Ai2020", "Chen2020", "Prada2013"]
    pybamm.set_logging_level("NOTICE")
    cycle_number = 1
    min_voltage = 2.6
    max_voltage = 4.1
    # exp = pybamm.Experiment(
    #     [(
    #         f"Discharge at 1C until {min_voltage} V",  # ageing cycles
    #         f"Charge at 0.3C until {max_voltage} V",
    #         f"Hold at {max_voltage} V until C/20",
    #         # "Rest for 4 hours",
    #     )] * cycle_number
    # )
    exp = pybamm.Experiment(
        [(
            f"Discharge at 1C until {min_voltage} V",  # ageing cycles
            # f"Charge at 0.3C until {max_voltage} V",
            # f"Hold at {min_voltage} V until C/20",
            # "Rest for 4 hours",
        )] * cycle_number
    )

    for i, value in enumerate(param_values):
        model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
        parameter_values = pybamm.ParameterValues(param_list[2])
        param_dict = {
            param_name: value,
            # experiment
            # "Lower voltage cut-off [V]": 2.5,
            # "Upper voltage cut-off [V]": 3.65,
            # cell
            # "Negative current collector thickness [m]": 0.000006,
            # "Separator thickness [m]": 0.000016,
            # "Positive current collector thickness [m]": 0.000013,
            # "Electrode height [m]": 0.325,
            # "Electrode width [m]": 10.3,
            # "Positive current collector density [kg.m-3]": 2700,
            # "Positive current collector thermal conductivity [W.m-1.K-1]": 237,
            # "Nominal cell capacity [A.h]": 100,
            # "Contact resistance [Ohm]": 0.0004,
            # # electrolyte
            # "Electrolyte conductivity [S.m-1]": 0.97,
        }
        # Update the parameter value
        parameter_values.update(param_dict, check_already_exists=False)
        # Create a simulation
        sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=exp)
        # Define the parameter to vary

        safe_solver = pybamm.CasadiSolver(mode="safe", dt_max=120)

        # Run the simulation
        sim.solve(solver=safe_solver, calc_esoh=False)

        # Extract the time and voltage
        time = sim.solution["Time [min]"].entries
        voltage = sim.solution["Voltage [V]"].entries

        # Plot voltage vs time
        ax.plot(time, voltage, label=f"{value:.1e} m", color=cmap(norm(value)))

    plt.xlabel('Time [min]')
    plt.ylabel('Terminal Voltage [V]')
    plt.title(param_name)
    plt.grid(True)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    plt.tight_layout()

    plt.show()
