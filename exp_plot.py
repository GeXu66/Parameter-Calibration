import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np

if __name__ == '__main__':
    matplotlib.use('TkAgg')
    # 定义文件名列表
    file_names = [
        'exp_81#-T25-0.1C.csv',
        'exp_81#-T25-0.2C.csv',
        'exp_81#-T25-0.33C.csv',
        'exp_81#-T25-1C.csv'
    ]
    method = "GA"
    directory_path = f"simu_data/{method}/"
    # 初始化一个空的DataFrame来存储所有数据
    all_data = pd.DataFrame()
    # 遍历文件名列表，读取每个文件
    for file_name in file_names:
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
    fig, ax = plt.subplots(1, 1)
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
                 # label=f'Real Voltage {experiment}',
                 linestyle='--', linewidth=2, color='black')
        # 绘制simu_voltage曲线（实线，线宽2）
        plt.plot(subset['real_time'], subset['simu_voltage'],
                 label=f'Simulation Voltage {experiment}',
                 linestyle='-', linewidth=2)  # 假设simu_voltage用蓝色表示
        # 计算RMSE
        real_voltage = subset_80['real_voltage'].values
        simu_voltage = subset_80['simu_voltage'].values
        mse = mean_squared_error(real_voltage, simu_voltage)
        rmse = np.sqrt(mse)
        rmse_values.append((experiment, rmse))
    # 添加图例
    plt.plot([], [], linestyle='--', linewidth=2, color='black', label='Real Discharge Voltage')
    plt.legend(loc='upper right')
    # 设置图表标题和轴标签
    plt.title('Real and Simulated Voltages Over Time')
    plt.xlabel('Real Time')
    plt.ylabel('Voltage')
    # 打印RMSE值
    for experiment, rmse in rmse_values:
        print(f'RMSE for {experiment}: {rmse:.4f}')
    # 显示图表
    plt.tight_layout()
    plt.show()
    fig.savefig(f"all_plot/{method}.png", dpi=300)
