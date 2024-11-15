import pybamm
import numpy as np
import os
from datetime import datetime
import pandas as pd
import SALib.sample.sobol as saltelli
from SALib.analyze import sobol
from SALib.analyze import pawn
import matplotlib.pyplot as plt


def min_max_func(min_val, max_val, x):
    """将0-1之间的归一化值转换回原始范围"""
    return min_val + x * (max_val - min_val)


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
    min_voltage = min_max_func(min_voltage - 1.5, min_voltage + 0.5, param[40])
    max_voltage = min_max_func(max_voltage - 0.5, max_voltage + 1.5, param[41])
    exp = pybamm.Experiment(
        [(
            f"Discharge at {discharge_cur} C for {time_max} seconds",  # ageing cycles
            # f"Discharge at 0.5 C until {min_voltage}V",  # ageing cycles
            # f"Charge at 0.5 C for 1830 seconds",  # ageing cycles
        )]
    )
    option = {"cell geometry": "arbitrary", "thermal": "lumped", "contact resistance": "false"}
    model = pybamm.lithium_ion.SPM()  # Doyle-Fuller-Newman model
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


def analyze_battery_performance(params):
    """运行单次电池仿真并返回性能指标"""
    try:
        # 设置固定参数
        min_voltage_base = 2.5  # 示例基准值，请根据实际情况调整
        max_voltage_base = 3.3  # 示例基准值，请根据实际情况调整
        discharge_cur = 1.0  # 放电倍率
        time_max = 3600  # 仿真时间
        capacity = 280  # 电池容量
        temperature = 25  # 温度

        # 运行仿真
        parameter_values, solution = pybamm_sim(
            params, min_voltage_base, max_voltage_base, discharge_cur, time_max, capacity, temperature
        )

        # 提取数据
        time = solution["Time [s]"].data
        voltage = solution["Terminal voltage [V]"].data

        # 电压特征
        mean_voltage = np.mean(voltage)
        voltage_decay = (voltage[0] - voltage[-1]) / time[-1]  # V/s
        voltage_stability = np.std(voltage)  # 电压标准差

        # 计算电压平台平坦度 (使用电压曲线的二阶导数)
        voltage_gradient = np.gradient(voltage, time)
        platform_flatness = np.mean(np.abs(np.gradient(voltage_gradient, time)))

        # 组合特征为综合性能指标
        # 对各个特征进行归一化和加权
        weights = {
            'mean_voltage': 0.5,
            'voltage_stability': 0.2,
            'platform_flatness': 0.3,
        }

        # 归一化处理（这里使用一些典型值作为参考）
        normalized_features = {
            'mean_voltage': mean_voltage / max_voltage_base,
            'voltage_stability': (voltage_stability / mean_voltage) * 100,  # 越稳定越好
            'platform_flatness': (platform_flatness * 1e6),  # 越平坦越好
        }
        for k in weights.keys():
            print(f'{k}:', normalized_features[k])

        # 计算综合性能指标
        performance_index = sum(weights[k] * normalized_features[k] for k in weights.keys())

        return performance_index

    except Exception as e:
        print(f"Simulation failed: {e}")
        return 0


def run_sensitivity_analysis(problem, param_values, Y):
    """执行敏感性分析并保存结果"""

    # 创建sensitivity文件夹(如果不存在)
    if not os.path.exists('sensitivity'):
        os.makedirs('sensitivity')

    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sobol分析
    Si_sobol = sobol.analyze(problem, Y, print_to_console=True, parallel=True, n_processors=16)
    # PAWN分析
    Si_pawn = pawn.analyze(problem, param_values, Y)

    # 创建Sobol结果DataFrame并排序
    results_sobol = pd.DataFrame({
        'Parameter': problem['names'],
        'First_Order': Si_sobol['S1'],
        'Total_Order': Si_sobol['ST'],
        'Confidence_S1': Si_sobol['S1_conf'],
        'Confidence_ST': Si_sobol['ST_conf']
    })
    # 按First_Order降序排序
    results_sobol = results_sobol.sort_values('First_Order', ascending=False)

    # 创建更详细的PAWN结果DataFrame并排序
    results_pawn = pd.DataFrame({
        'Parameter': problem['names'],
        'PAWN_minimum': Si_pawn['minimum'],
        'PAWN_mean': Si_pawn['mean'],
        'PAWN_median': Si_pawn['median'],
        'PAWN_maximum': Si_pawn['maximum'],
        'PAWN_CV': Si_pawn['CV']
    })
    # 按PAWN_median降序排序
    results_pawn = results_pawn.sort_values('PAWN_median', ascending=False)

    # 添加排名列
    results_sobol.insert(0, 'Rank', np.arange(1, len(results_sobol) + 1))
    results_pawn.insert(0, 'Rank', np.arange(1, len(results_pawn) + 1))

    # 保存结果到CSV
    results_sobol.to_csv(f'sensitivity/sobol_results_{timestamp}.csv', index=False)
    results_pawn.to_csv(f'sensitivity/pawn_results_{timestamp}.csv', index=False)

    # 创建比较结果的DataFrame
    comparison = pd.DataFrame()

    # 从Sobol结果中获取排序后的参数和对应的值
    sobol_data = results_sobol[['Parameter', 'First_Order']].copy()
    sobol_data['Sobol_Rank'] = range(1, len(sobol_data) + 1)

    # 从PAWN结果中获取排序后的参数和对应的值
    pawn_data = results_pawn[['Parameter', 'PAWN_median']].copy()
    pawn_data['PAWN_Rank'] = range(1, len(pawn_data) + 1)
    # 合并两个结果
    comparison = sobol_data.merge(pawn_data, on='Parameter', how='outer')
    # 重命名列
    comparison.columns = ['Parameter', 'Sobol_First_Order', 'Sobol_Rank', 'PAWN_Median', 'PAWN_Rank']
    # 按Sobol一阶指数降序排序
    comparison = comparison.sort_values('Sobol_First_Order', ascending=False)
    # 保存比较结果
    comparison.to_csv(f'sensitivity/comparison_results_{timestamp}.csv', index=False)

    # 绘制Sobol结果
    plt.figure(figsize=(15, 8))
    plt.bar(np.arange(len(problem['names'])), Si_sobol['S1'])
    plt.xticks(np.arange(len(problem['names'])), problem['names'], rotation=90)
    plt.title('First-order Sobol Sensitivity Indices')
    plt.xlabel('Parameters')
    plt.ylabel('Sensitivity Index')
    plt.tight_layout()
    plt.savefig(f'sensitivity/sobol_first_order_{timestamp}.png')
    plt.close()

    # 绘制更详细的PAWN结果
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # 绘制主要PAWN指标（中位数）
    ax1.bar(np.arange(len(problem['names'])), Si_pawn['median'])
    ax1.set_title('PAWN Median Sensitivity Indices')
    ax1.set_xlabel('Parameters')
    ax1.set_ylabel('Median KS Statistic')
    ax1.set_xticks(np.arange(len(problem['names'])))
    ax1.set_xticklabels(problem['names'], rotation=90)

    # 绘制所有PAWN统计量的箱线图
    data = [
        [Si_pawn['minimum'][i], Si_pawn['mean'][i],
         Si_pawn['median'][i], Si_pawn['maximum'][i]]
        for i in range(len(problem['names']))
    ]
    ax2.boxplot(data, tick_labels=problem['names'])
    ax2.set_title('PAWN Sensitivity Statistics Distribution')
    ax2.set_xlabel('Parameters')
    ax2.set_ylabel('KS Statistic')
    plt.xticks(rotation=90)

    plt.tight_layout()
    plt.savefig(f'sensitivity/pawn_detailed_{timestamp}.png')
    plt.close()

    return Si_sobol, Si_pawn


if __name__ == '__main__':
    # 定义参数空间
    # 创建sensitivity文件夹(如果不存在)
    if not os.path.exists('sensitivity'):
        os.makedirs('sensitivity')
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    problem = {
        'num_vars': 42,
        'names': [
            'electrode_height',
            'electrode_width',
            'Negative_electrode_conductivity',
            'Positive_electrode_diffusivity',
            'Positive_particle_radius',
            'Initial_concentration_in_positive_electrode',
            'Initial_concentration_in_negative_electrode',
            'Positive_electrode_conductivity',
            'Negative_particle_radius',
            'Negative_electrode_thickness',
            'Total_heat_transfer_coefficient',
            'Separator_density',
            'Separator_thermal_conductivity',
            'Positive_electrode_porosity',
            'Separator_specific_heat_capacity',
            'Maximum_concentration_in_positive_electrode',
            'Negative_electrode_Bruggeman_coefficient',
            'Positive_electrode_Bruggeman_coefficient',
            'Separator_porosity',
            'Negative_current_collector_thickness',
            'Positive_current_collector_thickness',
            'Positive_electrode_thickness',
            'Positive_electrode_active_material_volume_fraction',
            'Negative_electrode_specific_heat_capacity',
            'Positive_electrode_thermal_conductivity',
            'Negative_electrode_active_material_volume_fraction',
            'Negative_electrode_density',
            'Positive_electrode_specific_heat_capacity',
            'Positive_electrode_density',
            'Negative_electrode_thermal_conductivity',
            'Cation_transference_number',
            'Positive_current_collector_thermal_conductivity',
            'Negative_current_collector_thermal_conductivity',
            'Separator_Bruggeman_coefficient',
            'Maximum_concentration_in_negative_electrode',
            'Positive_current_collector_density',
            'Negative_current_collector_density',
            'Positive_current_collector_conductivity',
            'Negative_current_collector_conductivity',
            'Negative_electrode_porosity',
            'min_voltage',
            'max_voltage'
        ],
        'bounds': [[0, 1] for _ in range(42)]  # 所有参数都在0-1之间
    }
    # 主程序
    N = 512  # 样本数
    param_values = saltelli.sample(problem, N)

    # 执行模拟
    Y = np.zeros([param_values.shape[0]])
    for i, params in enumerate(param_values):
        print(f"Running simulation {i + 1}/{len(param_values)}")
        Y[i] = analyze_battery_performance(params)

    # 运行敏感性分析并保存结果
    Si_sobol, Si_pawn = run_sensitivity_analysis(problem, param_values, Y)

    # 打印排序后的Sobol结果
    print("\nParameters ranked by first-order Sobol sensitivity index:")
    sensitivity_results = list(zip(problem['names'], Si_sobol['S1'], Si_sobol['ST']))
    sensitivity_results.sort(key=lambda x: x[1], reverse=True)
    for name, S1, ST in sensitivity_results:
        print(f"{name:50s} First-order: {S1:.4f} Total-order: {ST:.4f}")

    # 打印详细的PAWN结果
    print("\nDetailed PAWN sensitivity analysis results:")
    pawn_results = list(zip(
        problem['names'],
        Si_pawn['minimum'],
        Si_pawn['mean'],
        Si_pawn['median'],
        Si_pawn['maximum'],
        Si_pawn['CV']
    ))
    pawn_results.sort(key=lambda x: x[3], reverse=True)  # 按中位数排序

    print("\nParameter | Min | Mean | Median | Max | CV")
    print("-" * 70)
    for name, min_val, mean_val, median_val, max_val, cv in pawn_results:
        print(f"{name:30s} | {min_val:.4f} | {mean_val:.4f} | {median_val:.4f} | {max_val:.4f} | {cv:.4f}")
