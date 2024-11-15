import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy import stats


@dataclass
class BoundaryHistory:
    """参数边界调整历史"""
    parameter_names: List[str]
    lower_bounds: List[List[float]]
    upper_bounds: List[List[float]]
    best_parameters: List[List[float]]
    fitness_values: List[float]


class AdaptiveBoundaryAdjuster:
    def __init__(self,
                 parameter_names: List[str],
                 initial_bounds: List[Tuple[float, float]],
                 min_bound_width: List[float],
                 max_bound_width: List[float],
                 history_window: int = 50):
        """
        自适应边界调整器
        Args:
            parameter_names: 参数名称列表
            initial_bounds: 初始参数边界 [(min1, max1), (min2, max2), ...]
            min_bound_width: 每个参数允许的最小边界宽度
            max_bound_width: 每个参数允许的最大边界宽度
            history_window: 历史窗口大小
        """
        self.parameter_names = parameter_names
        self.num_parameters = len(parameter_names)
        self.current_bounds = np.array(initial_bounds)
        self.min_bound_width = np.array(min_bound_width)
        self.max_bound_width = np.array(max_bound_width)
        self.history_window = history_window

        # 初始化历史记录
        self.history = BoundaryHistory(
            parameter_names=parameter_names,
            lower_bounds=[],
            upper_bounds=[],
            best_parameters=[],
            fitness_values=[]
        )

        # 参数分布统计
        self.param_stats = {
            'means': np.zeros(self.num_parameters),
            'stds': np.zeros(self.num_parameters),
            'success_rates': np.zeros(self.num_parameters),
            'exploration_scores': np.zeros(self.num_parameters)
        }

    def update_boundaries(self,
                          current_population: np.ndarray,
                          population_fitness: np.ndarray,
                          best_solution: np.ndarray,
                          best_fitness: float,
                          iteration: int) -> np.ndarray:
        """
        更新参数边界
        Args:
            current_population: 当前种群参数 [n_pop, n_params]
            population_fitness: 种群适应度值 [n_pop]
            best_solution: 当前最优解
            best_fitness: 当前最优适应度
            iteration: 当前迭代次数
        Returns:
            更新后的参数边界 [n_params, 2]
        """
        # 更新历史记录
        self._update_history(best_solution, best_fitness)

        # 更新参数统计信息
        self._update_parameter_statistics(current_population, population_fitness)

        # 计算边界调整因子
        adjustment_factors = self._calculate_adjustment_factors(iteration)

        # 应用边界调整
        new_bounds = self._adjust_boundaries(adjustment_factors)

        # 验证和规范化新边界
        self.current_bounds = self._validate_boundaries(new_bounds)

        return self.current_bounds

    def _update_history(self, best_solution: np.ndarray, best_fitness: float):
        """更新历史记录"""
        self.history.lower_bounds.append(self.current_bounds[:, 0].tolist())
        self.history.upper_bounds.append(self.current_bounds[:, 1].tolist())
        self.history.best_parameters.append(best_solution.tolist())
        self.history.fitness_values.append(best_fitness)

        # 维护历史窗口大小
        if len(self.history.fitness_values) > self.history_window:
            self.history.lower_bounds.pop(0)
            self.history.upper_bounds.pop(0)
            self.history.best_parameters.pop(0)
            self.history.fitness_values.pop(0)

    def _update_parameter_statistics(self,
                                     population: np.ndarray,
                                     fitness: np.ndarray):
        """更新参数统计信息"""
        # 计算参数分布
        self.param_stats['means'] = np.mean(population, axis=0)
        self.param_stats['stds'] = np.std(population, axis=0)

        # 计算参数成功率
        success_threshold = np.median(fitness)
        successful_solutions = population[fitness <= success_threshold]
        if len(successful_solutions) > 0:
            for i in range(self.num_parameters):
                param_range = self.current_bounds[i, 1] - self.current_bounds[i, 0]
                successful_range = (np.max(successful_solutions[:, i]) -
                                    np.min(successful_solutions[:, i]))
                self.param_stats['success_rates'][i] = successful_range / param_range

        # 计算探索得分
        for i in range(self.num_parameters):
            coverage = (np.max(population[:, i]) - np.min(population[:, i])) / \
                       (self.current_bounds[i, 1] - self.current_bounds[i, 0])
            diversity = stats.entropy(np.histogram(population[:, i], bins=10)[0])
            self.param_stats['exploration_scores'][i] = coverage * diversity

    def _calculate_adjustment_factors(self, iteration: int) -> np.ndarray:
        """计算边界调整因子"""
        adjustment_factors = np.ones(self.num_parameters)

        for i in range(self.num_parameters):
            # 1. 基于成功率的调整
            if self.param_stats['success_rates'][i] > 0.8:
                # 如果成功率高，缩小边界
                adjustment_factors[i] *= 0.9
            elif self.param_stats['success_rates'][i] < 0.2:
                # 如果成功率低，扩大边界
                adjustment_factors[i] *= 1.1

            # 2. 基于参数分布的调整
            mean_position = (self.param_stats['means'][i] - self.current_bounds[i, 0]) / \
                            (self.current_bounds[i, 1] - self.current_bounds[i, 0])
            if mean_position < 0.3 or mean_position > 0.7:
                # 如果最优解偏向边界，相应调整边界
                adjustment_factors[i] *= 1.1

            # 3. 基于探索得分的调整
            if self.param_stats['exploration_scores'][i] < 0.3:
                # 如果探索不足，扩大边界
                adjustment_factors[i] *= 1.2

            # 4. 基于优化阶段的调整
            stage_factor = max(0.5, 1 - iteration / 1000)  # 假设最大迭代次数为1000
            adjustment_factors[i] *= stage_factor

            # 5. 基于历史趋势的调整
            if len(self.history.fitness_values) > 2:
                recent_improvement = (self.history.fitness_values[-1] -
                                      self.history.fitness_values[-2])
                if recent_improvement > 0:
                    # 如果性能在改善，保持当前策略
                    adjustment_factors[i] = 1.0

        return adjustment_factors

    def _adjust_boundaries(self, adjustment_factors: np.ndarray) -> np.ndarray:
        """应用边界调整"""
        new_bounds = self.current_bounds.copy()

        for i in range(self.num_parameters):
            center = (self.current_bounds[i, 1] + self.current_bounds[i, 0]) / 2
            half_width = (self.current_bounds[i, 1] - self.current_bounds[i, 0]) / 2

            # 调整边界宽度
            new_half_width = half_width * adjustment_factors[i]

            # 更新边界
            new_bounds[i, 0] = center - new_half_width
            new_bounds[i, 1] = center + new_half_width

        return new_bounds

    def _validate_boundaries(self, bounds: np.ndarray) -> np.ndarray:
        """验证和规范化边界"""
        validated_bounds = bounds.copy()

        for i in range(self.num_parameters):
            width = bounds[i, 1] - bounds[i, 0]

            # 确保边界宽度在允许范围内
            if width < self.min_bound_width[i]:
                center = (bounds[i, 1] + bounds[i, 0]) / 2
                validated_bounds[i, 0] = center - self.min_bound_width[i] / 2
                validated_bounds[i, 1] = center + self.min_bound_width[i] / 2
            elif width > self.max_bound_width[i]:
                center = (bounds[i, 1] + bounds[i, 0]) / 2
                validated_bounds[i, 1] = center + self.max_bound_width[i] / 2
                validated_bounds[i, 0] = center - self.max_bound_width[i] / 2

        return validated_bounds

    def get_boundary_history(self) -> Dict:
        """获取边界调整历史"""
        return {
            'parameter_names': self.parameter_names,
            'lower_bounds': self.history.lower_bounds,
            'upper_bounds': self.history.upper_bounds,
            'best_parameters': self.history.best_parameters,
            'fitness_values': self.history.fitness_values
        }


def visualize_boundary_adaptation(boundary_history: Dict):
    """可视化边界调整过程"""
    import matplotlib.pyplot as plt

    num_params = len(boundary_history['parameter_names'])
    iterations = range(len(boundary_history['fitness_values']))

    # 绘制边界变化
    fig, axes = plt.subplots(num_params, 1, figsize=(10, 3 * num_params))
    for i, param_name in enumerate(boundary_history['parameter_names']):
        lower_bounds = [bounds[i] for bounds in boundary_history['lower_bounds']]
        upper_bounds = [bounds[i] for bounds in boundary_history['upper_bounds']]
        best_params = [params[i] for params in boundary_history['best_parameters']]

        axes[i].fill_between(iterations, lower_bounds, upper_bounds, alpha=0.3)
        axes[i].plot(iterations, best_params, 'r-', label='最优值')
        axes[i].set_title(f'参数 {param_name} 边界适应过程')
        axes[i].set_xlabel('迭代次数')
        axes[i].set_ylabel('参数值')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()