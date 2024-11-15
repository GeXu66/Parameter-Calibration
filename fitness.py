import matplotlib
import numpy as np
from scipy import signal, integrate
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import deque


@dataclass
class BatteryData:
    voltage: np.ndarray
    current: np.ndarray
    time: np.ndarray


@dataclass
class WeightHistory:
    """记录权重调整历史"""
    f1_weights: List[float] or deque
    f2_weights: List[float] or deque
    f3_weights: List[float] or deque
    fitness_values: List[float] or deque


class DynamicWeightAdjuster:
    def __init__(self,
                 initial_weights: List[float] = [0.4, 0.3, 0.3],
                 history_size: int = 50,
                 learning_rate: float = 0.05):
        """
        动态权重调整器
        Args:
            initial_weights: 初始权重 [w1, w2, w3]
            history_size: 历史记录大小
            learning_rate: 学习率
        """
        self.weights = np.array(initial_weights)
        self.history_size = history_size
        self.learning_rate = learning_rate

        # 初始化历史记录
        self.history = WeightHistory(
            f1_weights=deque(maxlen=history_size),
            f2_weights=deque(maxlen=history_size),
            f3_weights=deque(maxlen=history_size),
            fitness_values=deque(maxlen=history_size)
        )

        # 误差统计
        self.error_stats = {
            'f1_mean': 0.0,
            'f2_mean': 0.0,
            'f3_mean': 0.0,
            'f1_std': 0.0,
            'f2_std': 0.0,
            'f3_std': 0.0
        }

        self.iteration = 0

    def update_weights(self,
                       f1: float,
                       f2: float,
                       f3: float,
                       fitness: float) -> np.ndarray:
        """
        更新权重
        Args:
            f1, f2, f3: 当前迭代的三个误差项
            fitness: 当前适应度值
        Returns:
            更新后的权重数组
        """
        self.iteration += 1

        # 记录历史
        self.history.f1_weights.append(self.weights[0])
        self.history.f2_weights.append(self.weights[1])
        self.history.f3_weights.append(self.weights[2])
        self.history.fitness_values.append(fitness)

        # 计算误差统计量
        self._update_error_stats(f1, f2, f3)

        # 基于误差特征调整权重
        if self.iteration > self.history_size:
            self._adjust_weights_based_on_stats(f1, f2, f3)

        # 确保权重和为1
        self.weights = self._normalize_weights(self.weights)

        return self.weights

    def _update_error_stats(self, f1: float, f2: float, f3: float):
        """更新误差统计信息"""
        # 使用指数移动平均更新均值
        alpha = 0.1
        self.error_stats['f1_mean'] = (1 - alpha) * self.error_stats['f1_mean'] + alpha * f1
        self.error_stats['f2_mean'] = (1 - alpha) * self.error_stats['f2_mean'] + alpha * f2
        self.error_stats['f3_mean'] = (1 - alpha) * self.error_stats['f3_mean'] + alpha * f3

        # 更新标准差
        if self.iteration > 1:
            self.error_stats['f1_std'] = np.std([f1, self.error_stats['f1_mean']])
            self.error_stats['f2_std'] = np.std([f2, self.error_stats['f2_mean']])
            self.error_stats['f3_std'] = np.std([f3, self.error_stats['f3_mean']])

    def _adjust_weights_based_on_stats(self, f1: float, f2: float, f3: float):
        """基于误差统计特征调整权重"""
        # 计算相对误差
        total_error = f1 + f2 + f3
        if total_error == 0:
            return

        relative_errors = np.array([f1, f2, f3]) / total_error

        # 计算误差变化趋势
        error_trends = np.array([
            abs(f1 - self.error_stats['f1_mean']),
            abs(f2 - self.error_stats['f2_mean']),
            abs(f3 - self.error_stats['f3_mean'])
        ])

        # 计算误差波动性
        error_volatility = np.array([
            self.error_stats['f1_std'],
            self.error_stats['f2_std'],
            self.error_stats['f3_std']
        ])

        # 综合考虑多个因素调整权重
        weight_adjustments = np.zeros(3)

        # 1. 相对误差大的项增加权重
        weight_adjustments += relative_errors * 0.3

        # 2. 误差变化趋势大的项增加权重
        weight_adjustments += error_trends * 0.3

        # 3. 误差波动大的项减少权重
        weight_adjustments -= error_volatility * 0.2

        # 4. 考虑历史表现
        if len(self.history.fitness_values) >= 2:
            fitness_improvement = self.history.fitness_values[-1] - self.history.fitness_values[-2]
            if fitness_improvement < 0:  # 如果性能在改善
                # 增加贡献较大的权重
                contributing_weights = np.array([
                    self.history.f1_weights[-1],
                    self.history.f2_weights[-1],
                    self.history.f3_weights[-1]
                ])
                weight_adjustments += contributing_weights * 0.2

        # 应用权重调整
        self.weights += self.learning_rate * weight_adjustments

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """归一化权重，确保和为1且非负"""
        weights = np.clip(weights, 0.1, 0.8)  # 限制权重范围
        return weights / np.sum(weights)

    def get_weight_history(self) -> Dict:
        """获取权重调整历史"""
        return {
            'f1_weights': list(self.history.f1_weights),
            'f2_weights': list(self.history.f2_weights),
            'f3_weights': list(self.history.f3_weights),
            'fitness_values': list(self.history.fitness_values)
        }


class ParameterIdentification:
    def __init__(self):
        self.error_calculator = BatteryErrorCalculator()
        self.weight_adjuster = DynamicWeightAdjuster()

    def fitness_function(self,
                         parameters: List[float],
                         exp_data: BatteryData,
                         param_bounds: List[Tuple[float, float]]) -> float:
        """
        使用动态权重的适应度函数
        """
        # 将参数列表转换为字典
        param_dict = {
            'R0': parameters[0],
            'R1': parameters[1],
            'C1': parameters[2],
            'R2': parameters[3],
            'C2': parameters[4]
        }

        # 创建电池模型并仿真
        model = BatteryModel(param_dict)
        sim_voltage = model.simulate(exp_data.current, exp_data.time)

        # 构建仿真数据结构
        sim_data = BatteryData(
            voltage=sim_voltage,
            current=exp_data.current,
            time=exp_data.time
        )

        # 计算三个误差项
        f1 = self.error_calculator.calculate_curve_fitting_error(
            sim_voltage, exp_data.voltage)

        f2 = self.error_calculator.calculate_key_points_error(
            sim_data, exp_data)

        f3 = self.error_calculator.calculate_derivative_error(
            sim_data, exp_data)

        # 获取当前权重
        weights = self.weight_adjuster.weights

        # 计算加权适应度
        fitness = weights[0] * f1 + weights[1] * f2 + weights[2] * f3

        # 更新权重
        self.weight_adjuster.update_weights(f1, f2, f3, fitness)

        return fitness

    def optimize_parameters(self,
                            exp_data: BatteryData,
                            param_bounds: List[Tuple[float, float]],
                            max_iterations: int = 1000) -> Tuple[List[float], Dict]:
        """
        参数优化主函数
        Returns:
            最优参数和权重调整历史
        """
        # ... (优化算法实现，如PSO或遗传算法)

        # 返回优化结果和权重历史
        best_parameters = []  # 优化得到的最优参数
        weight_history = self.weight_adjuster.get_weight_history()

        return best_parameters, weight_history


def visualize_weight_history(weight_history: Dict):
    """
    可视化权重调整过程
    """
    import matplotlib.pyplot as plt

    iterations = np.arange(len(weight_history['f1_weights']))
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, weight_history['f1_weights'], label='w1 (曲线拟合)')
    plt.plot(iterations, weight_history['f2_weights'], label='w2 (关键点)')
    plt.plot(iterations, weight_history['f3_weights'], label='w3 (导数匹配)')

    plt.xlabel('迭代次数')
    plt.ylabel('权重值')
    plt.title('权重动态调整过程')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 绘制适应度变化
    plt.figure(figsize=(10, 4))
    plt.plot(iterations, weight_history['fitness_values'], label='适应度值')
    plt.xlabel('迭代次数')
    plt.ylabel('适应度值')
    plt.title('适应度收敛过程')
    plt.legend()
    plt.grid(True)
    plt.show()


class BatteryErrorCalculator:
    def __init__(self):
        self.sampling_rate = 1.0  # Hz

    def calculate_curve_fitting_error(self,
                                      sim_voltage: np.ndarray,
                                      exp_voltage: np.ndarray) -> float:
        """
        计算f1：整体曲线拟合误差
        使用RMSE(均方根误差)和MAE(平均绝对误差)的组合
        """
        # 确保数据长度一致
        min_len = min(len(sim_voltage), len(exp_voltage))
        sim_voltage = sim_voltage[:min_len]
        exp_voltage = exp_voltage[:min_len]

        # 计算RMSE
        rmse = np.sqrt(np.mean((sim_voltage - exp_voltage) ** 2))

        # 计算MAE
        mae = np.mean(np.abs(sim_voltage - exp_voltage))

        # 综合误差(可调整权重)
        error = 0.7 * rmse + 0.3 * mae

        return error

    def calculate_key_points_error(self,
                                   sim_data: BatteryData,
                                   exp_data: BatteryData) -> float:
        """
        计算f2：关键特征点误差
        包括：充电截止点、放电截止点、CC-CV切换点等
        """
        key_points_error = 0

        # 1. 找出CC-CV切换点
        def find_cc_cv_point(voltage: np.ndarray, current: np.ndarray) -> Tuple[int, float, float]:
            current_diff = np.abs(np.diff(current))
            cv_start_idx = np.where(current_diff > 0.1)[0][0]
            return cv_start_idx, voltage[cv_start_idx], current[cv_start_idx]

        # 计算实验和仿真的CC-CV切换点
        sim_cv_idx, sim_cv_v, sim_cv_i = find_cc_cv_point(sim_data.voltage, sim_data.current)
        exp_cv_idx, exp_cv_v, exp_cv_i = find_cc_cv_point(exp_data.voltage, exp_data.current)

        # 计算切换点误差
        cv_voltage_error = abs(sim_cv_v - exp_cv_v)
        cv_current_error = abs(sim_cv_i - exp_cv_i)

        # 2. 计算充放电截止点误差
        end_voltage_error = abs(sim_data.voltage[-1] - exp_data.voltage[-1])

        # 3. 计算电压平台区域误差
        def find_plateau_region(voltage: np.ndarray) -> Tuple[int, int]:
            voltage_diff = np.abs(np.diff(voltage))
            plateau_mask = voltage_diff < np.mean(voltage_diff) * 0.5
            plateau_start = np.where(plateau_mask)[0][0]
            plateau_end = np.where(plateau_mask)[0][-1]
            return plateau_start, plateau_end

        sim_plateau_start, sim_plateau_end = find_plateau_region(sim_data.voltage)
        exp_plateau_start, exp_plateau_end = find_plateau_region(exp_data.voltage)

        plateau_error = abs(sim_plateau_end - sim_plateau_start -
                            (exp_plateau_end - exp_plateau_start))

        # 综合所有特征点误差
        key_points_error = (0.4 * cv_voltage_error +
                            0.3 * end_voltage_error +
                            0.2 * cv_current_error +
                            0.1 * plateau_error)

        return key_points_error

    def calculate_derivative_error(self,
                                   sim_data: BatteryData,
                                   exp_data: BatteryData) -> float:
        """
        计算f3：导数匹配误差
        考虑电压变化率的匹配度
        """

        # 计算电压变化率
        def calculate_voltage_rate(voltage: np.ndarray, time: np.ndarray) -> np.ndarray:
            return np.gradient(voltage, time)

        # 计算实验和仿真的电压变化率
        sim_dv_dt = calculate_voltage_rate(sim_data.voltage, sim_data.time)
        exp_dv_dt = calculate_voltage_rate(exp_data.voltage, exp_data.time)

        # 计算变化率误差
        rate_error = np.mean(np.abs(sim_dv_dt - exp_dv_dt))

        # 计算二阶导数误差(加速度)
        sim_d2v_dt2 = np.gradient(sim_dv_dt, sim_data.time)
        exp_d2v_dt2 = np.gradient(exp_dv_dt, exp_data.time)

        acceleration_error = np.mean(np.abs(sim_d2v_dt2 - exp_d2v_dt2))

        # 综合导数误差
        derivative_error = 0.7 * rate_error + 0.3 * acceleration_error

        return derivative_error


class BatteryModel:
    def __init__(self, params: Dict[str, float]):
        """
        电池等效电路模型
        params包含: R0, R1, C1, R2, C2等参数
        """
        self.params = params

    def simulate(self, current: np.ndarray, time: np.ndarray) -> np.ndarray:
        """
        模拟电池响应
        这里使用二阶RC等效电路模型
        """
        dt = time[1] - time[0]
        n = len(time)

        # 初始化电压数组
        v_total = np.zeros(n)
        v1 = np.zeros(n)  # RC1的电压
        v2 = np.zeros(n)  # RC2的电压

        # OCV-SOC关系（简化）
        ocv = 3.7 * np.ones(n)  # 假设恒定OCV

        # 时域响应计算
        for i in range(1, n):
            # RC1响应
            v1[i] = v1[i - 1] * np.exp(-dt / (self.params['R1'] * self.params['C1'])) + \
                    self.params['R1'] * current[i] * (1 - np.exp(-dt / (self.params['R1'] * self.params['C1'])))

            # RC2响应
            v2[i] = v2[i - 1] * np.exp(-dt / (self.params['R2'] * self.params['C2'])) + \
                    self.params['R2'] * current[i] * (1 - np.exp(-dt / (self.params['R2'] * self.params['C2'])))

            # 总电压
            v_total[i] = ocv[i] - self.params['R0'] * current[i] - v1[i] - v2[i]

        return v_total


def example_usage():
    # 生成示例数据
    time = np.linspace(0, 1000, 1000)
    exp_current = np.zeros_like(time)
    exp_current[200:800] = 1.0  # 模拟恒流充电
    exp_voltage = 3.7 + 0.3 * np.sin(time / 100) + 0.1 * np.random.randn(len(time))

    # 创建实验数据结构
    exp_data = BatteryData(
        voltage=exp_voltage,
        current=exp_current,
        time=time
    )

    # 参数边界
    param_bounds = [
        (0.001, 0.1),  # R0
        (0.001, 0.1),  # R1
        (100, 1000),  # C1
        (0.001, 0.1),  # R2
        (1000, 10000)  # C2
    ]

    # 测试参数
    test_parameters = [0.05, 0.05, 500, 0.05, 5000]

    # 创建参数辨识对象并计算适应度
    identifier = ParameterIdentification()
    best_parameters, weight_history = identifier.optimize_parameters(exp_data, param_bounds)
    # 可视化权重调整过程
    visualize_weight_history(weight_history)


if __name__ == "__main__":
    matplotlib.use('TkAgg')
    example_usage()
