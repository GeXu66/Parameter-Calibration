import pybamm
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd
import pygad
import os
import json
import argparse
from scipy.interpolate import interp1d
import multiprocessing
from skopt import gp_minimize
from skopt.space import Real
from matplotlib import cm, colors, colormaps


def plot_time_vs_voltage(file_path, time_simulation, voltage_simulation):
    data = pd.read_csv(file_path)
    # 绘制SOC vs Voltage图
    fig, ax = plt.subplots()
    time = data['Test_Time(s)']
    voltage = data['Voltage(V)']
    ax.plot(time[:1830], voltage[:1830], linestyle='-')

    # voltage = voltage[:1830]
    num_points = 1830
    time_resampled = np.linspace(1, 1830, num_points)
    interp_func_sim = interp1d(time_simulation, voltage_simulation, kind='linear', fill_value="extrapolate")
    voltage_simulation_resampled = interp_func_sim(time_resampled)
    # z1 = np.polyfit(voltage_simulation_resampled, voltage, 3)  # 用7次多项式拟合，可改变多项式阶数；
    # p1 = np.poly1d(z1)  # 得到多项式系数，按照阶数从高到低排列
    # print(p1)  # 显示多项式
    # voltage_simulation = p1(voltage_simulation)  # 可直接使用yvals=np.polyval(z1,xxx)
    ax.plot(time_simulation, voltage_simulation, linestyle='-')

    plt.title('SOC vs Voltage')
    plt.xlabel('SOC')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.tight_layout()

    df = pd.DataFrame({"real_time": time[:1830], "real_voltage": voltage[:1830], "simu_time": time_resampled, "simu_voltage": voltage_simulation_resampled})
    df.to_csv("data.csv", index=False, sep=",")

    plt.show()


def plot_soc_vs_voltage_real_bat(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    # 绘制SOC vs Voltage图
    fig, ax = plt.subplots()
    soc = data['SOC'].str.rstrip('%').astype(float)
    voltage = data['V']
    ax.plot(soc, voltage, marker='o', linestyle='-')
    plt.title('SOC vs Voltage')
    plt.xlabel('SOC')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.tight_layout()


def plot_time_vs_voltage_real_bat(file_path):
    # 读取CSV文件
    data = pd.read_csv(file_path)
    # 绘制SOC vs Voltage图
    fig, ax = plt.subplots()
    ax.plot(data['time'], data['V'], marker='o', linestyle='-')
    plt.title('Time vs Voltage')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.tight_layout()


def compute_soc_discharge(sol, capacity, file_path):
    soc_init = 1
    # Extract the time and voltage
    soc_simulation = (soc_init - sol["Discharge capacity [A.h]"].entries / capacity) * 100
    voltage_simulation = sol["Voltage [V]"].entries

    # Resample voltage_simulation to 7200 points
    num_points = 720
    soc_resampled = np.linspace(0, 100, num_points)
    interp_func = interp1d(soc_simulation, voltage_simulation, kind='linear', fill_value="extrapolate")
    voltage_simulation_resampled = interp_func(soc_resampled)

    # real bat
    data = pd.read_csv(file_path)
    soc = data['SOC'].str.rstrip('%').astype(float)
    voltage = data['V']

    # Resample real voltage data to 7200 points
    interp_func_real = interp1d(soc, voltage, kind='linear', fill_value="extrapolate")
    voltage_resampled = interp_func_real(soc_resampled)

    # Calculate RMSE
    rmse_value = np.sqrt(np.mean((voltage_simulation_resampled - voltage_resampled) ** 2))
    print(f"SOC DISCHARGE RMSE: {rmse_value}")

    return soc_resampled, voltage_simulation_resampled, voltage_resampled, rmse_value


def plot_soc_discharge(soc_resampled, voltage_simulation_resampled, voltage_resampled, rmse_value):
    fig, ax = plt.subplots()
    ax.plot(soc_resampled, voltage_simulation_resampled, marker='o', linestyle='-', label='Simulation')
    ax.plot(soc_resampled, voltage_resampled, marker='o', linestyle='-', label='Experiment')
    plt.xlabel('State of Charge [%]')
    plt.ylabel('Terminal Voltage [V]')
    plt.title(f"SOC vs Terminal Voltage--RMSE:{rmse_value:.4f} V")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def compute_time_discharge(sol, file_path):
    # Extract time and voltage from the simulation
    time_simulation = sol["Time [s]"].entries
    voltage_simulation = sol["Voltage [V]"].entries
    # Load real battery data
    data = pd.read_csv(file_path)
    time_real = data['time']
    voltage_real = data['V']
    time_max = data['time'].values[-1]
    simu_time_max = time_simulation[-1]
    real_time_max = min(time_max, simu_time_max)

    # Resample voltage_simulation to 7200 points
    time_resampled = np.arange(0, real_time_max + 1, 10)
    interp_func_sim = interp1d(time_simulation, voltage_simulation, kind='linear', fill_value="extrapolate")
    voltage_simulation_resampled = interp_func_sim(time_resampled)

    # Resample real voltage data to 7200 points
    interp_func_real = interp1d(time_real, voltage_real, kind='linear', fill_value="extrapolate")
    voltage_real_resampled = interp_func_real(time_resampled)

    # Calculate RMSE
    rmse_value = np.sqrt(mean_squared_error(voltage_real_resampled, voltage_simulation_resampled))
    print(f"TIME DISCHARGE RMSE: {rmse_value}")
    return time_resampled, voltage_simulation_resampled, voltage_real_resampled, rmse_value


def plot_time_discharge(time_resampled, voltage_simulation_resampled, voltage_real_resampled, rmse_value, discharge_cur):
    fig, ax = plt.subplots()
    # Plotting
    ax.plot(time_resampled, voltage_simulation_resampled, linestyle='-', label='Simulation')
    ax.plot(time_resampled, voltage_real_resampled, linestyle='-', label='Experiment')

    plt.xlabel('Time [s]')
    plt.ylabel('Terminal Voltage [V]')
    plt.title(f"{file_name}-{discharge_cur}C-RMSE:{rmse_value:.4f} V")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"./simu_fig/{subdir_name}/{file_name}_{discharge_cur}C.png")
    plt.show()


def read_file(file_name):
    data = pd.read_csv(file_name)
    time_max = data['time'].values[-1]
    voltage_max = data['V'].values[0]
    voltage_min = data['V'].values[-1] - 1
    capacity = data['Ah'].values[-1]
    return time_max, voltage_max, voltage_min, capacity


def pybamm_sim(param, min_voltage, max_voltage, discharge_cur, time_max, capacity, temperature):
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
    Negative_electrode_density = min_max_func(1555, 3100, param[26])
    Positive_electrode_specific_heat_capacity = min_max_func(700, 1270, param[27])
    Positive_electrode_density = min_max_func(2341, 4206, param[28])
    Negative_electrode_thermal_conductivity = min_max_func(1.04, 1.7, param[29])
    Cation_transference_number = min_max_func(0.25, 0.4, param[30])
    Positive_current_collector_thermal_conductivity = min_max_func(158, 238, param[31])
    Negative_current_collector_thermal_conductivity = min_max_func(267, 401, param[32])
    Separator_Bruggeman_coefficient = min_max_func(1.5, 2, param[33])
    Maximum_concentration_in_negative_electrode = min_max_func(24983, 33133, param[34])
    Positive_current_collector_density = min_max_func(2700, 3490, param[35])
    Negative_current_collector_density = min_max_func(8933, 11544, param[36])
    Positive_current_collector_conductivity = min_max_func(35500000, 37800000, param[37])
    Negative_current_collector_conductivity = min_max_func(58411000, 59600000, param[38])
    Negative_electrode_porosity = min_max_func(0.25, 0.5, param[39])
    min_voltage = min_max_func(min_voltage - 0.3, min_voltage + 0.3, param[40])
    max_voltage = min_max_func(max_voltage - 0.3, max_voltage + 0.3, param[41])
    exp = pybamm.Experiment(
        [(
            f"Discharge at {discharge_cur} C for {time_max} seconds",  # ageing cycles
            # f"Discharge at 0.5 C until {min_voltage}V",  # ageing cycles
            # f"Charge at 0.5 C for 1830 seconds",  # ageing cycles
        )]
    )
    option = {"cell geometry": "arbitrary", "thermal": "lumped", "contact resistance": "false"}
    if model_type == "DFN":
        model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
    else:
        model = pybamm.lithium_ion.SPM()

    parameter_values = pybamm.ParameterValues("Prada2013")
    param_dict = {
        "Number of electrodes connected in parallel to make a cell": 1,
        "Nominal cell capacity [A.h]": capacity,
        "Lower voltage cut-off [V]": min_voltage,
        "Upper voltage cut-off [V]": max_voltage,
        "Ambient temperature [K]": 273.15 + 25,
        "Initial temperature [K]": 273.15 + temperature,
        # "Total heat transfer coefficient [W.m-2.K-1]": 10,
        # "Cell cooling surface area [m2]": 0.126,
        # "Cell volume [m3]": 0.00257839,
        # cell
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
        "Negative electrode density [kg.m-3]": Negative_electrode_density,
        "Positive electrode specific heat capacity [J.kg-1.K-1]": Positive_electrode_specific_heat_capacity,
        "Positive electrode density [kg.m-3]": Positive_electrode_density,
        "Negative electrode thermal conductivity [W.m-1.K-1]": Negative_electrode_thermal_conductivity,
        "Cation transference number": Cation_transference_number,
        "Positive current collector thermal conductivity [W.m-1.K-1]": Positive_current_collector_thermal_conductivity,
        "Negative current collector thermal conductivity [W.m-1.K-1]": Negative_current_collector_thermal_conductivity,
        "Separator Bruggeman coefficient (electrolyte)": Separator_Bruggeman_coefficient,
        "Maximum concentration in negative electrode [mol.m-3]": Maximum_concentration_in_negative_electrode,
        "Positive current collector density [kg.m-3]": Positive_current_collector_density,
        "Negative current collector density [kg.m-3]": Negative_current_collector_density,
        "Positive current collector conductivity [S.m-1]": Positive_current_collector_conductivity,
        "Negative current collector conductivity [S.m-1]": Negative_current_collector_conductivity,
        "Negative electrode porosity": Negative_electrode_porosity,

    }
    # Update the parameter value
    parameter_values.update(param_dict, check_already_exists=False)
    # Create a simulation
    sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=exp)
    # Define the parameter to vary
    safe_solver = pybamm.CasadiSolver(mode="safe", dt_max=120)
    # Run the simulation
    sim.solve(solver=safe_solver, calc_esoh=False, initial_soc=1)
    sol = sim.solution
    return parameter_values, sol


def main_simulationMO(param, save=False, plot=False):
    param_list = ["Ai2020", "Chen2020", "Prada2013"]
    # pybamm.set_logging_level("NOTICE")
    names = name.split(",")
    file_list = [f"./bat_data/{single}.csv" for single in names]
    all_time_rmse = []
    for i, file in enumerate(file_list):
        discharge_cur = float(names[i].split("-")[-1].replace("C", ""))
        temperature = int(names[i].split("-")[1].replace("T", ""))
        time_max, voltage_max, voltage_min, capacity = read_file(file_name=file)
        min_voltage = voltage_min
        max_voltage = voltage_max
        parameter_values, sol = pybamm_sim(param, min_voltage, max_voltage, discharge_cur, time_max, capacity, temperature)
        # soc_resampled, soc_voltage_simulation_resampled, soc_voltage_resampled, soc_rmse_value = compute_soc_discharge(sol=sol, capacity=parameter_values["Nominal cell capacity [A.h]"],file_path=file)
        time_resampled, time_voltage_simulation_resampled, time_voltage_resampled, time_rmse_value = compute_time_discharge(sol=sol, file_path=file)
        all_time_rmse.append(time_rmse_value)
        if plot:
            # plot_soc_discharge(soc_resampled, soc_voltage_simulation_resampled, soc_voltage_resampled, soc_rmse_value)
            plot_time_discharge(time_resampled, time_voltage_simulation_resampled, time_voltage_resampled, time_rmse_value, discharge_cur)
        if save:
            df = pd.DataFrame({"real_time": time_resampled, "real_voltage": time_voltage_resampled, "simu_time": time_resampled, "simu_voltage": time_voltage_simulation_resampled})
            df.to_csv(f"./simu_data/{subdir_name}/exp_{file_name}-T{temperature}-{discharge_cur}C-{model_type}.csv", index=False, sep=",")
    return all_time_rmse


def catch_error_simulation(solution, return_dict):
    try:
        all_time_rmse = main_simulationMO(solution)
        return_dict['result'] = all_time_rmse  # 返回计算结果
    except Exception as e:
        print(f"Error occurred: {e}")
        return_dict['result'] = [100] * wc_num  # 发生错误时设置为 None


# 定义包装函数以处理超时和错误
def run_with_timeout(param, timeout=15):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()  # 用于在进程间共享数据
    process = multiprocessing.Process(target=catch_error_simulation, args=(param, return_dict))
    process.start()
    process.join(timeout)  # 等待进程完成，最多等待 timeout 秒

    if process.is_alive():
        process.terminate()  # 超时，终止进程
        process.join()  # 等待进程真正结束
        return [100] * wc_num  # 超时返回 100
    # 检查返回值是否为 NaN
    result = return_dict.get('result')
    try:
        print("\033[31m result:\033[0m", result)
        if True in np.isnan(result):
            return [100] * wc_num  # 发生错误或返回 NaN，返回 100
        else:
            return result  # 返回正常结果
    except Exception as e:
        print(f"Error occurred: {e}")
        return [100] * wc_num


def fitness_func(ga_instance, solution, solution_idx):
    all_time_rmse = run_with_timeout(solution)
    fitness = [-time_rmse_value for time_rmse_value in all_time_rmse]
    print("\033[31m Norm Solution Value\033[0m", solution)
    # fitness = -np.log(time_rmse_value)
    print("\033[31m RMSE (mV):\033[0m", [mv * 1000 for mv in all_time_rmse])
    print("\033[31m fitness:\033[0m", fitness)
    return fitness


def obj_func(solution):
    time_rmse_value = run_with_timeout(solution)
    fitness = time_rmse_value ** 2
    print("Norm Solution Value", solution)
    # fitness = -np.log(time_rmse_value)
    print("RMSE (mV):", time_rmse_value * 1000)
    return fitness


def on_generation(ga_instance):
    global last_fitness
    print(f"\033[31m Generation = {ga_instance.generations_completed}\033[0m")
    print(f"\033[31m Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}\033[0m")
    print(f"\033[31m Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}\033[0m")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


def min_max_func(low, high, norm_value):
    return norm_value * (high - low) + low


def ga_optimization(file_name):
    num_genes = 42
    num_generations = 300  # Number of generations.
    num_parents_mating = 20  # Number of solutions to be selected as parents in the mating pool.
    sol_per_pop = 30  # Number of solutions in the population.
    # define gene space
    gene_space = [
        {'low': 0, 'high': 1},  # Electrode height
        {'low': 0, 'high': 1},  # Electrode width
        {'low': 0, 'high': 1},  # Negative electrode conductivity
        {'low': 0, 'high': 1},  # Positive electrode diffusivity
        {'low': 0, 'high': 1},  # Positive particle radius
        {'low': 0, 'high': 1},  # Initial concentration in positive electrode
        {'low': 0, 'high': 1},  # Initial concentration in negative electrode
        {'low': 0, 'high': 1},  # Positive electrode conductivity
        {'low': 0, 'high': 1},  # Negative particle radius
        {'low': 0, 'high': 1},  # Negative electrode thickness
        {'low': 0, 'high': 1},  # Total heat transfer coefficient
        {'low': 0, 'high': 1},  # Separator density
        {'low': 0, 'high': 1},  # Separator thermal conductivity
        {'low': 0, 'high': 1},  # Positive electrode porosity
        {'low': 0, 'high': 1},  # Separator specific heat capacity
        {'low': 0, 'high': 1},  # Maximum concentration in positive electrode
        {'low': 0, 'high': 1},  # Negative electrode Bruggeman coefficient
        {'low': 0, 'high': 1},  # Positive electrode Bruggeman coefficient
        {'low': 0, 'high': 1},  # Separator porosity
        {'low': 0, 'high': 1},  # Negative current collector thickness
        {'low': 0, 'high': 1},  # Positive current collector thickness
        {'low': 0, 'high': 1},  # Positive electrode thickness
        {'low': 0, 'high': 1},  # Positive electrode active material volume fraction
        {'low': 0, 'high': 1},  # Negative electrode specific heat capacity
        {'low': 0, 'high': 1},  # Positive electrode thermal conductivity
        {'low': 0, 'high': 1},  # Negative electrode active material volume fraction
        {'low': 0, 'high': 1},  # Negative electrode density
        {'low': 0, 'high': 1},  # Positive_electrode_specific_heat_capacity
        {'low': 0, 'high': 1},  # Positive_electrode_density
        {'low': 0, 'high': 1},  # Negative_electrode_thermal_conductivity
        {'low': 0, 'high': 1},  # Cation_transference_number
        {'low': 0, 'high': 1},  # Positive_current_collector_thermal_conductivity
        {'low': 0, 'high': 1},  # Negative_current_collector_thermal_conductivity
        {'low': 0, 'high': 1},  # Separator_Bruggeman_coefficient
        {'low': 0, 'high': 1},  # Maximum_concentration_in_negative_electrode
        {'low': 0, 'high': 1},  # Positive_current_collector_density
        {'low': 0, 'high': 1},  # Negative_current_collector_density
        {'low': 0, 'high': 1},  # Positive_current_collector_conductivity
        {'low': 0, 'high': 1},  # Negative_current_collector_conductivity
        {'low': 0, 'high': 1},  # Negative_electrode_porosity
        {'low': 0, 'high': 1},  # Min voltage
        {'low': 0, 'high': 1},  # Max voltage
    ]

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           save_best_solutions=True,
                           fitness_func=fitness_func,
                           on_generation=on_generation,
                           parallel_processing=16,
                           parent_selection_type='nsga2')

    # Running the GA to optimize the parameters of the function.
    ga_instance.run()
    ga_instance.plot_fitness(label=['Obj 1', 'Obj 2', 'Obj 3', 'Obj 4'])

    pareto_fronts = ga_instance.pareto_fronts
    for front_level in pareto_fronts:
        for solution, fitness in front_level:
            print(f"Solution: {solution}, Fitness: {fitness}")

    best_solution = ga_instance.population[0]
    print("best solution:", best_solution)

    # Returning the details of the best solution.
    # solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    # print(f"Parameters of the best solution : {solution}")
    # print(f"Fitness value of the best solution = {solution_fitness}")
    # print(f"Index of the best solution : {solution_idx}")

    # Saving the GA instance.
    filename = f'./solutions/{subdir_name}/{file_name}'  # The filename to which the instance is saved. The name is without extension.
    ga_instance.save(filename=filename)

    prediction = main_simulationMO(best_solution, save=True, plot=True)
    print(f"Predicted output based on the best solution : {prediction}")

    if ga_instance.best_solution_generation != -1:
        print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

    output_data = {
        "best_parameters": best_solution,
        "best_rmse_value": prediction,
    }

    with open(f"./solutions/{subdir_name}/{file_name}.json", "w") as f:
        json.dump(output_data, f, cls=NumpyEncoder)

    # Loading the saved GA instance.
    # loaded_ga_instance = pygad.load(filename=filename)
    # loaded_ga_instance.plot_fitness()


def bayes_optimization(file_name):
    space = [Real(0, 1) for _ in range(42)]
    # 运行贝叶斯优化
    result = gp_minimize(
        func=obj_func,  # 目标函数
        dimensions=space,  # 搜索空间
        acq_func="gp_hedge",
        n_calls=500,  # 优化迭代次数
        random_state=42,  # 随机种子
        n_jobs=-1,
    )
    # 输出结果
    print("最佳参数值:", result.x)
    print("最小RMSE:", result.fun)
    prediction = main_simulationMO(result.x, save=True, plot=True)
    print(f"Predicted output based on the best solution : {prediction}")
    output_data = {
        "best_parameters": result.x,
        "best_function_value": result.fun,
    }

    with open(f"./solutions/{subdir_name}/{file_name}.json", "w") as f:
        json.dump(output_data, f, cls=NumpyEncoder)


def local_optimization(file_name):
    space = [(0, 1) for _ in range(42)]  # 定义搜索空间，这里假设每个维度有42个可能的值
    initial_guess = np.random.rand(42)  # 随机生成一个初始猜测值

    # 运行局部优化
    result = minimize(
        fun=obj_func,  # 目标函数
        x0=initial_guess,  # 初始猜测值
        method='trust-constr',  # 使用L-BFGS-B算法
        bounds=space,  # 搜索空间的边界
        options={'maxiter': 10, 'disp': True},  # 设置最大迭代次数和显示进度
    )

    # 输出结果
    print("最佳参数值:", result.x)
    print("最小RMSE:", result.fun)
    prediction = main_simulationMO(result.x, save=True, plot=True)
    print(f"Predicted output based on the best solution : {prediction}")

    output_data = {
        "best_parameters": result.x,
        "best_function_value": result.fun,
    }

    with open(f"./solutions/{subdir_name}/{file_name}.json", "w") as f:
        json.dump(output_data, f, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    name_list = ["81#-T25-0.1C", "81#-T25-0.2C", "81#-T25-0.33C", "81#-T25-1C"]
    last_fitness = 0
    parser = argparse.ArgumentParser(description="Run GA optimization or load solution.")
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--filename', type=str, required=True, help='Filename for the optimization or solution.')
    parser.add_argument('--method', type=str, choices=["GA", "Bayes", "Local"], required=True, help='Optimization Method.')
    parser.add_argument('--model', type=str, choices=["DFN", "SPM"], required=True, help='Model Type.')
    args = parser.parse_args()
    name = args.filename
    model_type = args.model
    file_name = name.split(",")[0].split("-")[0] + "MO" + f"-{model_type}"
    subdir_name = args.method
    wc_num = len(args.filename.split(","))
    if args.train:
        if args.method == "GA":
            ga_optimization(file_name=file_name)
        elif args.method == "Local":
            local_optimization(file_name=file_name)
        else:
            bayes_optimization(file_name=file_name)
    else:
        sol_name = f'./solutions/{args.method}/{file_name}'
        loaded_ga_instance = pygad.load(filename=sol_name)
        solution, solution_fitness, solution_idx = loaded_ga_instance.best_solution(loaded_ga_instance.last_generation_fitness)
        print("best solution:", solution)
        # main_simulationMO(solution, save=True, plot=True)
        pareto_fronts = loaded_ga_instance.pareto_fronts
        for front_level in pareto_fronts:
            for solution, fitness in front_level:
                print(f"Solution idx: {solution}, Fitness: {fitness}")
        pareto_front = pareto_fronts[0]  # 假设我们关注的是第一个Pareto front
        # 遍历Pareto front，获取解决方案的索引和适应度
        for solution_idx, fitness in pareto_front:
            # 使用解决方案的索引来检索解决方案的实际数值
            solution = loaded_ga_instance.population[solution_idx]
            print(f"Solution Index: {solution_idx}, Solution: {solution}, Fitness: {fitness}")
            if all(abs(i) < 0.03 for i in fitness):
                main_simulationMO(solution, save=True, plot=True)
