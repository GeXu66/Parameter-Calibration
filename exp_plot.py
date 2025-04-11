import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import re


def draw_single_condition(method, file_name):
    directory_path = f"simu_data/{method}/"
    # 初始化一个空的DataFrame来存储所有数据
    all_data = pd.DataFrame()
    # 遍历文件名列表，读取每个文件
    # 读取CSV文件
    data = pd.read_csv(directory_path + file_name)
    # 将文件名添加为新列，用于区分不同的数据集
    data['experiment'] = file_name.split('-')[2]  # 假设文件名格式是固定的，并且实验编号在第三个'-'后面
    # 将数据添加到总的DataFrame中
    all_data = pd.concat([all_data, data], ignore_index=True)
    # 绘制图表
    # 计算RMSE
    rmse_values = []
    # 遍历每个唯一的实验编号，绘制每个实验的曲线
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=13)
    for experiment in all_data['experiment'].unique():
        subset = all_data[all_data['experiment'] == experiment]
        # 计算90%的数据点位置
        n_start = int(len(subset) * 0.2)
        n_end = int(len(subset) * 0.8)
        # 截取0到90%的数据
        subset_80 = subset.iloc[n_start:n_end]
        # 绘制图表
        # 绘制real_voltage曲线（黑虚线，线宽2）
        plt.plot(subset['real_time'], subset['real_voltage'],
                 label=f'Real Discharge Voltage under {experiment.replace(".csv", "")}',
                 linestyle='-', marker='v', linewidth=2, color='#D95319', markevery=len(subset) // 50)
        # 绘制simu_voltage曲线（实线，线宽2）
        plt.plot(subset['real_time'], subset['simu_voltage'],
                 label=f'Simulation Discharge Voltage under {experiment.replace(".csv", "")}',
                 linestyle='-', marker='^', linewidth=2, color='#0072BD', markevery=len(subset) // 50)  # 假设simu_voltage用蓝色表示
        # 计算RMSE
        real_voltage = subset_80['real_voltage'].values
        simu_voltage = subset_80['simu_voltage'].values
        mse = mean_squared_error(real_voltage, simu_voltage)
        rmse = np.sqrt(mse)
        rmse_values.append((experiment, rmse))
        plt.ylim(min(subset['simu_voltage']), max(subset['real_voltage']))
    # 添加图例
    # plt.plot([], [], linestyle='--', linewidth=2, color='black', label='Real Discharge Voltage')
    plt.legend(loc='lower left', fontsize=13)
    # 设置图表标题和轴标签
    # plt.title('Real and Simulated Voltages Over Time')
    plt.xlabel('Time (seconds)', fontsize=13)
    plt.ylabel('Voltage (V)', fontsize=13)
    # 打印RMSE值
    for experiment, rmse in rmse_values:
        print(f'RMSE for {experiment}: {rmse:.4f}')
    plt.show()
    file_name = file_name.replace(".csv", "")
    fig.savefig(f"all_plot/single/{method}-{file_name}.png", dpi=300)


def extract_pattern(filename):
    # 正则表达式模式，匹配数字后跟C
    pattern = r'-(\d+\.?\d*C)-'
    # 使用search方法查找匹配的字符串
    match = re.search(pattern, filename)
    if match:
        # 返回匹配的字符串
        return match.group(1)
    else:
        # 如果没有匹配，返回None
        return None


def draw_multi_condition(method, file_names):
    # 定义文件名列表
    directory_path = f"simu_data/{method}/"
    # 初始化一个空的DataFrame来存储所有数据
    all_data = pd.DataFrame()
    # 遍历文件名列表，读取每个文件
    for file_name in file_names:
        # 读取CSV文件
        data = pd.read_csv(directory_path + file_name)
        # 将文件名添加为新列，用于区分不同的数据集
        # data['experiment'] = file_name.split('-')[2]  # 假设文件名格式是固定的，并且实验编号在第三个'-'后面
        data['experiment'] = extract_pattern(file_name)
        # 将数据添加到总的DataFrame中
        all_data = pd.concat([all_data, data], ignore_index=True)
    # 绘制图表
    # 计算RMSE
    rmse_values = []
    # 遍历每个唯一的实验编号，绘制每个实验的曲线
    fig, ax = plt.subplots(figsize=(6.5, 4), constrained_layout=True)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True, labelsize=13)
    real_mark_list = ['v', '<', '+', '8']
    simu_mark_list = ['^', '>', 'x', 's']
    i = 0
    plot_min = 100
    for experiment in all_data['experiment'].unique():
        subset = all_data[all_data['experiment'] == experiment]
        # 计算90%的数据点位置
        n_start = int(len(subset) * 0.2)
        n_end = int(len(subset) * 0.8)
        # 截取0到90%的数据
        subset_80 = subset.iloc[n_start:n_end]
        # 绘制图表
        # 绘制real_voltage曲线（黑虚线，线宽2）
        plt.plot(subset['real_time'], subset['real_voltage'],
                 label=f'Real Voltage {experiment.replace(".csv", "")}',
                 linestyle='-', marker=real_mark_list[i], linewidth=2, color='#D95319', markevery=len(subset) // 50)
        # 绘制simu_voltage曲线（实线，线宽2）
        plt.plot(subset['real_time'], subset['simu_voltage'],
                 label=f'Simulation Voltage {experiment.replace(".csv", "")}',
                 linestyle='-', marker=simu_mark_list[i], linewidth=2, color='#0072BD', markevery=len(subset) // 50)  # 假设simu_voltage用蓝色表示
        # 计算RMSE
        real_voltage = subset_80['real_voltage'].values
        simu_voltage = subset_80['simu_voltage'].values
        mse = mean_squared_error(real_voltage, simu_voltage)
        rmse = np.sqrt(mse)
        rmse_values.append((experiment, rmse))
        plt.ylim(2.7, max(subset['real_voltage']))
        i = i + 1
    # 添加图例
    # plt.plot([], [], linestyle='--', linewidth=2, color='black', label='Real Discharge Voltage')
    plt.legend(loc='lower left', ncol=2, fontsize=12)
    # 设置图表标题和轴标签
    # plt.title('Real and Simulated Voltages Over Time')
    plt.xlabel('Time (seconds)', fontsize=13)
    plt.ylabel('Voltage (V)', fontsize=13)
    # 打印RMSE值
    for experiment, rmse in rmse_values:
        print(f'RMSE for {experiment}: {rmse:.4f}')
    # 显示图表
    plt.show()
    file = file_names[0].split('-')[0]
    fig.savefig(f"all_plot/multi/{method}-{file}.png", dpi=300)


if __name__ == '__main__':
    matplotlib.use('TkAgg')
    plt.rcParams['font.family'] = 'Times New Roman'
    file_names = [
        # 'exp_81#-T25-0.1C.csv',
        # 'exp_81#-T25-0.2C.csv',
        # 'exp_81#-T25-0.33C.csv',
        # 'exp_81#-T25-1C.csv'
        # 'exp_81#MO-T25-0.1C.csv',
        # 'exp_81#MO-T25-0.2C.csv',
        # 'exp_81#MO-T25-0.33C.csv',
        # 'exp_81#MO-T25-1.0C.csv'
        # 'exp_81#MO-DFN-T25-0.1C-DFN.csv',
        # 'exp_81#MO-DFN-T25-0.2C-DFN.csv',
        # 'exp_81#MO-DFN-T25-0.33C-DFN.csv',
        # 'exp_81#MO-DFN-T25-1.0C-DFN.csv'
        'exp_81#MO-Constraint-DFN-T25-0.1C-DFN.csv',
        'exp_81#MO-Constraint-DFN-T25-0.2C-DFN.csv',
        'exp_81#MO-Constraint-DFN-T25-0.33C-DFN.csv',
        'exp_81#MO-Constraint-DFN-T25-1.0C-DFN.csv'
    ]
    method = 'Bayes'
    draw_single_condition(method=method, file_name=file_names[0])
    draw_single_condition(method=method, file_name=file_names[1])
    draw_single_condition(method=method, file_name=file_names[2])
    draw_single_condition(method=method, file_name=file_names[3])
    draw_multi_condition(method=method, file_names=file_names)
