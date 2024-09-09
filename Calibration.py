import pybamm
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pygad
from scipy.interpolate import interp1d
import multiprocessing
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


def plot_time_discharge(time_resampled, voltage_simulation_resampled, voltage_real_resampled, rmse_value, name):
    fig, ax = plt.subplots()
    # Plotting
    ax.plot(time_resampled, voltage_simulation_resampled, linestyle='-', label='Simulation')
    ax.plot(time_resampled, voltage_real_resampled, linestyle='-', label='Experiment')

    plt.xlabel('Time [s]')
    plt.ylabel('Terminal Voltage [V]')
    plt.title(f"{name}--RMSE:{rmse_value:.4f} V")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def read_file(file_name):
    data = pd.read_csv(file_name)
    time_max = data['time'].values[-1]
    voltage_max = data['V'].values[0]
    voltage_min = data['V'].values[-1]
    capacity = data['Ah'].values[-1]
    return time_max, voltage_max, voltage_min, capacity


def main_simulation(param, save=False):
    electrode_height = param[0]
    electrode_width = param[1]
    print("electrode_height:", electrode_height)
    print("electrode_width", electrode_width)
    param_list = ["Ai2020", "Chen2020", "Prada2013"]
    pybamm.set_logging_level("NOTICE")
    name = "81#-T25-1C"
    file = f"./bat_data/{name}.csv"
    charge_capacity = float(name.split("-")[-1].replace("C", ""))
    temperature = int(name.split("-")[1].replace("T", ""))
    time_max, voltage_max, voltage_min, capacity = read_file(file_name=file)
    cycle_number = 1
    min_voltage = voltage_min
    # 3.3107
    max_voltage = voltage_max
    exp = pybamm.Experiment(
        [(
            f"Discharge at {charge_capacity} C for {time_max} seconds",  # ageing cycles
            # f"Discharge at 0.5 C until {min_voltage}V",  # ageing cycles
            # f"Charge at 0.5 C for 1830 seconds",  # ageing cycles
        )] * cycle_number
    )
    option = {"cell geometry": "arbitrary", "thermal": "lumped", "contact resistance": "false"}
    model = pybamm.lithium_ion.DFN()  # Doyle-Fuller-Newman model
    parameter_values = pybamm.ParameterValues(param_list[2])
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
        "Negative current collector thickness [m]": 0.00001,
        "Electrode height [m]": electrode_height,
        "Electrode width [m]": electrode_width,
    }
    # Update the parameter value
    parameter_values.update(param_dict, check_already_exists=False)
    # Create a simulation
    sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=exp)
    # Define the parameter to vary
    safe_solver = pybamm.CasadiSolver(mode="safe", dt_max=0.1)
    # Run the simulation
    sim.solve(solver=safe_solver, calc_esoh=False, initial_soc=1)
    sol = sim.solution
    soc_resampled, soc_voltage_simulation_resampled, soc_voltage_resampled, soc_rmse_value = compute_soc_discharge(sol=sol, capacity=parameter_values["Nominal cell capacity [A.h]"], file_path=file)
    time_resampled, time_voltage_simulation_resampled, time_voltage_resampled, time_rmse_value = compute_time_discharge(sol=sol, file_path=file)
    # plot_soc_discharge(soc_resampled, soc_voltage_simulation_resampled, soc_voltage_resampled, soc_rmse_value)
    plot_time_discharge(time_resampled, time_voltage_simulation_resampled, time_voltage_resampled, time_rmse_value, name)
    if save:
        df = pd.DataFrame({"real_time": time_resampled, "real_voltage": time_voltage_resampled, "simu_time": time_resampled, "simu_voltage": time_voltage_simulation_resampled})
        df.to_csv(f"./simu_data/exp_{name}.csv", index=False, sep=",")
    # soc_init = 1
    # # Extract the time and voltage
    # soc_simulation = (soc_init - sol["Discharge capacity [A.h]"].entries / 280) * 100
    # output_variables = [
    #     "Voltage [V]",
    #     "X-averaged cell temperature [K]",
    #     "Cell temperature [K]",
    #     "Resistance [Ohm]",
    # ]
    # fig, ax = plt.subplots()
    # ax.plot(soc_simulation)
    # pybamm.dynamic_plot(sol, output_variables)
    # time_simulation = sol["Time [s]"].entries
    # voltage_simulation = sol["Voltage [V]"].entries
    # plot_time_vs_voltage("./Huaiwei_data/data.csv", time_simulation, voltage_simulation)

    return time_rmse_value


def catch_error_simulation(solution, return_dict):
    try:
        time_rmse_value = main_simulation(solution)
        return_dict['result'] = time_rmse_value  # 返回计算结果
    except Exception as e:
        print(f"Error occurred: {e}")
        return_dict['result'] = None  # 发生错误时设置为 None


# 定义包装函数以处理超时和错误
def run_with_timeout(param, timeout=60):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()  # 用于在进程间共享数据
    process = multiprocessing.Process(target=catch_error_simulation, args=(param, return_dict))
    process.start()
    process.join(timeout)  # 等待进程完成，最多等待 timeout 秒

    if process.is_alive():
        process.terminate()  # 超时，终止进程
        process.join()  # 等待进程真正结束
        return 100  # 超时返回 100
    # 检查返回值是否为 NaN
    result = return_dict.get('result')
    if result is None or np.isnan(result):
        return 100  # 发生错误或返回 NaN，返回 100
    else:
        return result  # 返回正常结果


def fitness_func(ga_instance, solution, solution_idx):
    print(solution)
    time_rmse_value = run_with_timeout(solution)
    fitness = 1.0 / time_rmse_value
    print("fitness:", fitness)
    return fitness


def on_generation(ga_instance):
    global last_fitness
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")
    print(f"Change     = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1] - last_fitness}")
    last_fitness = ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]


def ga_optimization():
    num_genes = 2
    num_generations = 200  # Number of generations.
    num_parents_mating = 20  # Number of solutions to be selected as parents in the mating pool.
    sol_per_pop = 20  # Number of solutions in the population.
    # define gene space
    gene_space = [
        {'low': 0.6, 'high': 0.9},
        {'low': 27, 'high': 29}
    ]

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           gene_space=gene_space,
                           fitness_func=fitness_func,
                           on_generation=on_generation)

    # Running the GA to optimize the parameters of the function.
    ga_instance.run()

    ga_instance.plot_fitness()

    # Returning the details of the best solution.
    solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)
    print(f"Parameters of the best solution : {solution}")
    print(f"Fitness value of the best solution = {solution_fitness}")
    print(f"Index of the best solution : {solution_idx}")

    prediction = main_simulation(solution)
    print(f"Predicted output based on the best solution : {prediction}")

    if ga_instance.best_solution_generation != -1:
        print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

    # Saving the GA instance.
    filename = 'genetic'  # The filename to which the instance is saved. The name is without extension.
    ga_instance.save(filename=filename)

    # Loading the saved GA instance.
    loaded_ga_instance = pygad.load(filename=filename)
    loaded_ga_instance.plot_fitness()


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    last_fitness = 0
    # ga_optimization()
    sol = [0.725, 28]
    # sol = [0.86941868, 29.00751672]
    main_simulation(sol, save=True)
