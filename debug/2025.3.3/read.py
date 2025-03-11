# feasible_solutions.csv
# Condition 1 Best RMSE: 0.046015236968139674
# Condition 2 Best RMSE: 1.9421435764685306
# Condition 3 Best RMSE: 1.0967508584811836
# Condition 4 Best RMSE: 0.42685103645161476

# feasible_solutions2.csv


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.interpolate import interp1d

# 读取可行解文件
file_path = 'feasible_solutions2.csv'
data = pd.read_csv(file_path)


# 读取实际放电曲线数据
def read_actual_discharge_curves(file_paths):
    actual_curves = []
    for file_path in file_paths:
        data = pd.read_csv(file_path)
        time = data['time'].values
        voltage = data['V'].values
        actual_curves.append((time, voltage))
    return actual_curves


# 仿真放电曲线数据
def simulate_discharge_curves(params, actual_curves):
    simulated_curves = []
    for param, (time, voltage) in zip(params, actual_curves):
        interp_func = interp1d(time, voltage, kind='linear', fill_value="extrapolate")
        simulated_voltage = interp_func(time) * param  # 这里假设仿真数据与参数相关
        simulated_curves.append((time, simulated_voltage))
    return simulated_curves


# 计算RMSE
def calculate_rmse(actual, simulated):
    return np.sqrt(mean_squared_error(actual, simulated))


# 绘制放电曲线
def plot_discharge_curves_separately(actual_curves, simulated_curves, title_suffix=''):
    plt.figure(figsize=(12, 8))
    for i in range(len(actual_curves)):
        plt.subplot(2, 2, i + 1)
        actual_time, actual_voltage = actual_curves[i]
        simulated_time, simulated_voltage = simulated_curves[i]
        plt.plot(actual_time, actual_voltage, label=f'Actual Condition {i + 1}')
        plt.plot(simulated_time, simulated_voltage, label=f'Simulated Condition {i + 1}')
        plt.title(f'Condition {i + 1} {title_suffix}')
        plt.xlabel('Time')
        plt.ylabel('Voltage')
        plt.legend()
    plt.tight_layout()
    plt.show()


# 假设实际放电曲线文件路径
actual_discharge_files = [
    './bat_data/81#-T25-0.1C.csv',
    './bat_data/81#-T25-0.2C.csv',
    './bat_data/81#-T25-0.33C.csv',
    './bat_data/81#-T25-1C.csv'
]

# 读取实际放电曲线数据
actual_discharge_curves = read_actual_discharge_curves(actual_discharge_files)

# 计算每个可行解的RMSE
best_rmse_values = [float('inf')] * 4
best_solutions = [None] * 4
best_simulated_curves = [None] * 4

for index, row in data.iterrows():
    params = np.fromstring(row['Solution'].strip('[]'), sep=' ')
    simulated_curves = simulate_discharge_curves(params, actual_discharge_curves)
    for i in range(4):
        actual_time, actual_voltage = actual_discharge_curves[i]
        simulated_time, simulated_voltage = simulated_curves[i]
        rmse = calculate_rmse(actual_voltage, simulated_voltage)
        if rmse < best_rmse_values[i]:
            best_rmse_values[i] = rmse
            best_solutions[i] = params
            best_simulated_curves[i] = simulated_curves[i]

# 打印每个工况下的最优解的RMSE
for i in range(4):
    print(f'Condition {i + 1} Best RMSE: {best_rmse_values[i]}')

# 绘制所有工况下的最优解的放电曲线
plot_discharge_curves_separately(actual_discharge_curves, best_simulated_curves, title_suffix='Best RMSE')

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_squared_error
#
# # 读取可行解文件
# file_path = "feasible_solutions.csv"
# data = pd.read_csv(file_path)
#
# # 假设实际放电曲线和仿真放电曲线数据存储在以下变量中
# # 这里用随机数据代替，实际使用时请替换为真实数据
# actual_discharge_curves = [np.random.rand(100) for _ in range(4)]
# simulated_discharge_curves = [np.random.rand(100) for _ in range(4)]
#
# # 计算RMSE
# def calculate_rmse(actual, simulated):
#     return np.sqrt(mean_squared_error(actual, simulated))
#
# # 计算每个可行解的RMSE
# rmse_values = []
# for index, row in data.iterrows():
#     rmse = []
#     for i in range(4):
#         rmse.append(calculate_rmse(actual_discharge_curves[i], simulated_discharge_curves[i]))
#     rmse_values.append(np.mean(rmse))
#
# # 找到最优解
# best_solution_index = np.argmin(rmse_values)
# best_solution = data.iloc[best_solution_index]
#
# # 绘制最优解的放电曲线
# plt.figure(figsize=(12, 8))
# for i in range(4):
#     plt.subplot(2, 2, i+1)
#     plt.plot(actual_discharge_curves[i], label='Actual Discharge Curve')
#     plt.plot(simulated_discharge_curves[i], label='Simulated Discharge Curve')
#     plt.title(f'Condition {i+1}')
#     plt.xlabel('Time')
#     plt.ylabel('Voltage')
#     plt.legend()
#
# plt.tight_layout()
# plt.show()
#
# # 打印最优解的RMSE
# print(f'Best Solution RMSE: {rmse_values[best_solution_index]}')
