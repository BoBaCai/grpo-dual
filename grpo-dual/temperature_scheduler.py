#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temperature Scheduler for GRPO Multi-Objective Training

基于 DeepSeek-R1 和 EDT 的最佳实践：
- Stage-wise 降温（高探索 → 收敛 → 部署对齐）
- Per-task 差异化温度（Fairness vs Hallucination）
- 轻量自适应（熵 + 截断率驱动）

参考文献：
- DeepSeek-R1: Stage 1 T=1.0, Stage 2 T=0.7
- EDT: 熵驱动动态温度
- DAPO: 多目标 RL 长度控制
"""

from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class TemperatureConfig:
    """温度调度配置"""

    # 全局范围
    T_min: float = 0.6
    T_max: float = 1.3
    delta_T: float = 0.05  # 自适应步长

    # 熵目标
    entropy_target_low: float = 3.0
    entropy_target_high: float = 4.0

    # 截断率阈值（per-stage）
    trunc_threshold_stage1: float = 0.40
    trunc_threshold_stage2: float = 0.15
    trunc_threshold_stage3: float = 0.10

    # Stage 划分（比例）
    stage1_end: float = 0.30  # 0-30%: 探索期
    stage2_end: float = 0.80  # 30-80%: 收敛期
    # 80-100%: 部署对齐期

    # Per-task 基础温度（Stage 1）
    fairness_T_init: float = 1.10
    hallucination_T_init: float = 0.95

    # Per-task 温度范围（Stage 1）
    fairness_T_range_s1: Tuple[float, float] = (1.0, 1.25)
    hallucination_T_range_s1: Tuple[float, float] = (0.8, 1.10)

    # Per-task 温度目标（Stage 2 终点）
    fairness_T_end_s2: float = 0.90
    hallucination_T_end_s2: float = 0.80

    # Per-task 温度范围（Stage 2）
    fairness_T_range_s2: Tuple[float, float] = (0.8, 1.10)
    hallucination_T_range_s2: Tuple[float, float] = (0.7, 0.95)

    # Per-task 温度范围（Stage 3）
    fairness_T_range_s3: Tuple[float, float] = (0.75, 0.90)
    hallucination_T_range_s3: Tuple[float, float] = (0.70, 0.80)

    # 自适应模式（per-stage）
    stage1_adapt_mode: str = "truncation_only"  # "truncation_only", "entropy_only", "both", "none"
    stage2_adapt_mode: str = "both"
    stage3_adapt_mode: str = "truncation_only"

    # 统计窗口
    window_steps: int = 50  # 每 N 步更新一次温度


class TemperatureScheduler:
    """
    三阶段温度调度器，支持 per-task 和轻量自适应

    使用示例：
    ```python
    scheduler = TemperatureScheduler(total_steps=500)

    # 每个训练步
    temps = scheduler.get_temperature(
        step=current_step,
        fairness_entropy=2.5,
        fairness_trunc_rate=0.3,
        hallucination_entropy=3.2,
        hallucination_trunc_rate=0.2
    )

    T_fairness = temps['fairness']
    T_hallucination = temps['hallucination']
    ```
    """

    def __init__(self, total_steps: int, config: Optional[TemperatureConfig] = None):
        self.total_steps = total_steps
        self.config = config or TemperatureConfig()

        # 当前温度（per-task）
        self.current_T = {
            'fairness': self.config.fairness_T_init,
            'hallucination': self.config.hallucination_T_init
        }

        # 统计缓冲（用于滑动平均）
        self.entropy_buffer = {'fairness': [], 'hallucination': []}
        self.trunc_buffer = {'fairness': [], 'hallucination': []}

        # 历史记录（用于分析和可视化）
        self.history = {
            'step': [],
            'stage': [],
            'fairness_T': [],
            'hallucination_T': [],
            'fairness_entropy': [],
            'hallucination_entropy': [],
            'fairness_trunc': [],
            'hallucination_trunc': [],
            'fairness_adapt_reason': [],
            'hallucination_adapt_reason': []
        }

    def get_current_stage(self, step: int) -> int:
        """确定当前所处的 stage (1, 2, 3)"""
        progress = step / self.total_steps

        if progress <= self.config.stage1_end:
            return 1
        elif progress <= self.config.stage2_end:
            return 2
        else:
            return 3

    def get_stage_progress(self, step: int) -> float:
        """获取当前 stage 内的进度 [0.0, 1.0]"""
        progress = step / self.total_steps
        stage = self.get_current_stage(step)

        if stage == 1:
            return progress / self.config.stage1_end
        elif stage == 2:
            stage_start = self.config.stage1_end
            stage_length = self.config.stage2_end - self.config.stage1_end
            return (progress - stage_start) / stage_length
        else:  # stage == 3
            stage_start = self.config.stage2_end
            stage_length = 1.0 - self.config.stage2_end
            return (progress - stage_start) / stage_length

    def get_base_temperature(self, step: int, task: str) -> float:
        """
        获取基础温度（stage-wise schedule，不考虑自适应）

        Args:
            step: 当前步数
            task: 'fairness' or 'hallucination'

        Returns:
            基础温度值
        """
        stage = self.get_current_stage(step)
        stage_progress = self.get_stage_progress(step)

        if task == 'fairness':
            if stage == 1:
                return self.config.fairness_T_init
            elif stage == 2:
                # 线性退火从 Stage 1 末尾到 Stage 2 目标
                T_start = self.config.fairness_T_init
                T_end = self.config.fairness_T_end_s2
                return T_start + (T_end - T_start) * stage_progress
            else:  # stage == 3
                # 保持在 Stage 2 的终点值
                return self.config.fairness_T_end_s2

        else:  # hallucination
            if stage == 1:
                return self.config.hallucination_T_init
            elif stage == 2:
                T_start = self.config.hallucination_T_init
                T_end = self.config.hallucination_T_end_s2
                return T_start + (T_end - T_start) * stage_progress
            else:  # stage == 3
                return self.config.hallucination_T_end_s2

    def get_temperature_range(self, step: int, task: str) -> Tuple[float, float]:
        """获取当前 stage 下该任务的温度范围"""
        stage = self.get_current_stage(step)

        if task == 'fairness':
            if stage == 1:
                return self.config.fairness_T_range_s1
            elif stage == 2:
                return self.config.fairness_T_range_s2
            else:
                return self.config.fairness_T_range_s3
        else:  # hallucination
            if stage == 1:
                return self.config.hallucination_T_range_s1
            elif stage == 2:
                return self.config.hallucination_T_range_s2
            else:
                return self.config.hallucination_T_range_s3

    def get_adapt_mode(self, step: int) -> str:
        """获取当前 stage 的自适应模式"""
        stage = self.get_current_stage(step)

        if stage == 1:
            return self.config.stage1_adapt_mode
        elif stage == 2:
            return self.config.stage2_adapt_mode
        else:
            return self.config.stage3_adapt_mode

    def get_truncation_threshold(self, step: int) -> float:
        """获取当前 stage 的截断率阈值"""
        stage = self.get_current_stage(step)

        if stage == 1:
            return self.config.trunc_threshold_stage1
        elif stage == 2:
            return self.config.trunc_threshold_stage2
        else:
            return self.config.trunc_threshold_stage3

    def update_temperature_adaptive(
        self,
        task: str,
        entropy: float,
        trunc_rate: float,
        step: int
    ) -> Tuple[float, str]:
        """
        自适应调整温度（熵 + 截断率驱动）

        Args:
            task: 'fairness' or 'hallucination'
            entropy: 当前批次的平均熵
            trunc_rate: 当前批次的截断率
            step: 当前步数

        Returns:
            (新温度, 调整原因)
        """
        current_T = self.current_T[task]
        T_min, T_max = self.get_temperature_range(step, task)
        adapt_mode = self.get_adapt_mode(step)
        trunc_threshold = self.get_truncation_threshold(step)

        # 如果不启用自适应，直接返回基础温度
        if adapt_mode == "none":
            base_T = self.get_base_temperature(step, task)
            return np.clip(base_T, T_min, T_max), "none"

        # 初始化调整
        new_T = current_T
        reason = "stable"

        # 检查截断率
        if adapt_mode in ["truncation_only", "both"]:
            if trunc_rate > trunc_threshold:
                new_T = max(new_T - self.config.delta_T, T_min)
                reason = f"trunc_high({trunc_rate:.2f}>{trunc_threshold:.2f})"

        # 检查熵
        if adapt_mode in ["entropy_only", "both"] and reason == "stable":
            if entropy < self.config.entropy_target_low:
                new_T = min(new_T + self.config.delta_T, T_max)
                reason = f"entropy_low({entropy:.2f}<{self.config.entropy_target_low})"
            elif entropy > self.config.entropy_target_high:
                new_T = max(new_T - self.config.delta_T, T_min)
                reason = f"entropy_high({entropy:.2f}>{self.config.entropy_target_high})"

        # Clip 到允许范围
        new_T = max(T_min, min(T_max, new_T))

        return new_T, reason

    def get_temperature(
        self,
        step: int,
        fairness_entropy: Optional[float] = None,
        fairness_trunc_rate: Optional[float] = None,
        hallucination_entropy: Optional[float] = None,
        hallucination_trunc_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        获取当前步的温度（主接口）

        Args:
            step: 当前训练步数
            fairness_entropy: Fairness 任务的平均熵（可选，用于自适应）
            fairness_trunc_rate: Fairness 任务的截断率（可选）
            hallucination_entropy: Hallucination 任务的平均熵（可选）
            hallucination_trunc_rate: Hallucination 任务的截断率（可选）

        Returns:
            {'fairness': T_f, 'hallucination': T_h, 'stage': stage}
        """
        stage = self.get_current_stage(step)

        # 只在窗口边界更新温度（减少抖动）
        should_update = (step % self.config.window_steps == 0) or (step == 0)

        if should_update:
            # 更新 Fairness 温度
            if fairness_entropy is not None and fairness_trunc_rate is not None:
                new_T_f, reason_f = self.update_temperature_adaptive(
                    'fairness', fairness_entropy, fairness_trunc_rate, step
                )
                self.current_T['fairness'] = new_T_f
            else:
                # 如果没有提供指标，使用基础温度
                self.current_T['fairness'] = self.get_base_temperature(step, 'fairness')
                reason_f = "no_metrics"

            # 更新 Hallucination 温度
            if hallucination_entropy is not None and hallucination_trunc_rate is not None:
                new_T_h, reason_h = self.update_temperature_adaptive(
                    'hallucination', hallucination_entropy, hallucination_trunc_rate, step
                )
                self.current_T['hallucination'] = new_T_h
            else:
                self.current_T['hallucination'] = self.get_base_temperature(step, 'hallucination')
                reason_h = "no_metrics"

            # 记录历史
            self.history['step'].append(step)
            self.history['stage'].append(stage)
            self.history['fairness_T'].append(self.current_T['fairness'])
            self.history['hallucination_T'].append(self.current_T['hallucination'])
            self.history['fairness_entropy'].append(fairness_entropy or 0.0)
            self.history['hallucination_entropy'].append(hallucination_entropy or 0.0)
            self.history['fairness_trunc'].append(fairness_trunc_rate or 0.0)
            self.history['hallucination_trunc'].append(hallucination_trunc_rate or 0.0)
            self.history['fairness_adapt_reason'].append(reason_f)
            self.history['hallucination_adapt_reason'].append(reason_h)

            # 打印调试信息
            if step % (self.config.window_steps * 5) == 0:  # 每 5 个窗口打印一次
                f_ent_str = f"{fairness_entropy:.2f}" if fairness_entropy is not None else "N/A"
                f_trunc_str = f"{fairness_trunc_rate:.2%}" if fairness_trunc_rate is not None else "N/A"
                h_ent_str = f"{hallucination_entropy:.2f}" if hallucination_entropy is not None else "N/A"
                h_trunc_str = f"{hallucination_trunc_rate:.2%}" if hallucination_trunc_rate is not None else "N/A"

                print(f"\n🌡️ [Step {step}] Temperature Update (Stage {stage}):")
                print(f"  Fairness:      T={self.current_T['fairness']:.3f} | "
                      f"Entropy={f_ent_str} | Trunc={f_trunc_str} | Reason: {reason_f}")
                print(f"  Hallucination: T={self.current_T['hallucination']:.3f} | "
                      f"Entropy={h_ent_str} | Trunc={h_trunc_str} | Reason: {reason_h}")

        return {
            'fairness': self.current_T['fairness'],
            'hallucination': self.current_T['hallucination'],
            'stage': stage
        }

    def get_kl_coefficient(self, step: int) -> float:
        """
        获取当前步的 KL 系数（配合温度调度）

        参考 DeepSeek-R1: Stage 1 小 KL (0.001) → Stage 2-3 逐步增大
        """
        stage = self.get_current_stage(step)
        stage_progress = self.get_stage_progress(step)

        if stage == 1:
            return 0.003  # 低约束，高探索
        elif stage == 2:
            # 从 0.003 线性增长到 0.01
            return 0.003 + (0.01 - 0.003) * stage_progress
        else:  # stage == 3
            # 从 0.01 增长到 0.02
            return 0.01 + (0.02 - 0.01) * stage_progress

    def get_max_new_tokens(self, step: int) -> int:
        """
        获取当前步的 max_new_tokens（配合温度调度）

        Stage 1-2 前期: 256（给足空间）
        Stage 2 后期: 降到 192
        Stage 3: 保持 192
        """
        stage = self.get_current_stage(step)
        stage_progress = self.get_stage_progress(step)

        if stage == 1:
            return 256
        elif stage == 2:
            if stage_progress < 0.5:
                return 256
            else:
                # 线性从 256 降到 192
                return int(256 - (256 - 192) * (stage_progress - 0.5) / 0.5)
        else:  # stage == 3
            return 192

    def get_truncation_penalty(self, step: int) -> float:
        """
        获取截断惩罚系数（乘到 reward 上）

        Stage 1: 轻微 (0.7)
        Stage 2: 中等 (0.5)
        Stage 3: 严重 (0.3)
        """
        stage = self.get_current_stage(step)

        if stage == 1:
            return 0.7
        elif stage == 2:
            return 0.5
        else:
            return 0.3

    def get_length_penalty_lambda(self, step: int) -> float:
        """
        获取长度正则化系数

        Stage 1: 很小 (0.01)
        Stage 2: 中等 (0.03)
        Stage 3: 较大 (0.05)
        """
        stage = self.get_current_stage(step)

        if stage == 1:
            return 0.01
        elif stage == 2:
            return 0.03
        else:
            return 0.05

    def save_history(self, path: str):
        """保存温度调整历史到 CSV"""
        import csv

        if not self.history['step']:
            print("⚠️ No history to save")
            return

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)
            # 写入表头
            headers = list(self.history.keys())
            writer.writerow(headers)

            # 写入数据行
            num_rows = len(self.history['step'])
            for i in range(num_rows):
                row = [self.history[key][i] for key in headers]
                writer.writerow(row)

        print(f"✅ Temperature history saved to {path}")

    def plot_history(self, save_path: str = "temperature_history.png"):
        """绘制温度调整历史"""
        try:
            import matplotlib.pyplot as plt

            if not self.history['step']:
                print("⚠️ No history to plot")
                return

            # 直接使用 history 字典
            df = self.history

            fig, axes = plt.subplots(3, 2, figsize=(14, 10))
            fig.suptitle('Temperature Scheduler History', fontsize=16)

            # 温度曲线
            axes[0, 0].plot(df['step'], df['fairness_T'], label='Fairness', color='blue')
            axes[0, 0].plot(df['step'], df['hallucination_T'], label='Hallucination', color='red')
            axes[0, 0].set_ylabel('Temperature')
            axes[0, 0].set_title('Temperature Over Time')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 熵曲线
            axes[0, 1].plot(df['step'], df['fairness_entropy'], label='Fairness', color='blue')
            axes[0, 1].plot(df['step'], df['hallucination_entropy'], label='Hallucination', color='red')
            axes[0, 1].axhline(y=self.config.entropy_target_low, color='green', linestyle='--', alpha=0.5)
            axes[0, 1].axhline(y=self.config.entropy_target_high, color='orange', linestyle='--', alpha=0.5)
            axes[0, 1].set_ylabel('Entropy')
            axes[0, 1].set_title('Entropy Over Time')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

            # 截断率曲线
            axes[1, 0].plot(df['step'], df['fairness_trunc'], label='Fairness', color='blue')
            axes[1, 0].plot(df['step'], df['hallucination_trunc'], label='Hallucination', color='red')
            axes[1, 0].set_ylabel('Truncation Rate')
            axes[1, 0].set_title('Truncation Rate Over Time')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Stage 分布
            axes[1, 1].scatter(df['step'], df['stage'], alpha=0.5)
            axes[1, 1].set_ylabel('Stage')
            axes[1, 1].set_title('Training Stage')
            axes[1, 1].set_yticks([1, 2, 3])
            axes[1, 1].grid(True, alpha=0.3)

            # 温度 vs 熵（Fairness）
            axes[2, 0].scatter(df['fairness_entropy'], df['fairness_T'],
                              c=df['step'], cmap='viridis', alpha=0.6)
            axes[2, 0].set_xlabel('Entropy')
            axes[2, 0].set_ylabel('Temperature')
            axes[2, 0].set_title('Fairness: T vs Entropy (color=step)')
            axes[2, 0].grid(True, alpha=0.3)

            # 温度 vs 截断率（Hallucination）
            axes[2, 1].scatter(df['hallucination_trunc'], df['hallucination_T'],
                              c=df['step'], cmap='viridis', alpha=0.6)
            axes[2, 1].set_xlabel('Truncation Rate')
            axes[2, 1].set_ylabel('Temperature')
            axes[2, 1].set_title('Hallucination: T vs Truncation (color=step)')
            axes[2, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150)
            print(f"✅ Temperature plot saved to {save_path}")

        except ImportError:
            print("⚠️ matplotlib not available, skip plotting")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建调度器
    scheduler = TemperatureScheduler(total_steps=500)

    print("=" * 80)
    print("Temperature Scheduler Demo")
    print("=" * 80)

    # 模拟训练过程
    for step in [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
        # 模拟指标（随机）
        fairness_entropy = np.random.uniform(2.0, 4.5)
        fairness_trunc = np.random.uniform(0.1, 0.5)
        halu_entropy = np.random.uniform(2.5, 4.0)
        halu_trunc = np.random.uniform(0.05, 0.3)

        temps = scheduler.get_temperature(
            step=step,
            fairness_entropy=fairness_entropy,
            fairness_trunc_rate=fairness_trunc,
            hallucination_entropy=halu_entropy,
            hallucination_trunc_rate=halu_trunc
        )

        if step % 100 == 0:
            print(f"\nStep {step} (Stage {temps['stage']}):")
            print(f"  KL coef: {scheduler.get_kl_coefficient(step):.4f}")
            print(f"  Max tokens: {scheduler.get_max_new_tokens(step)}")
            print(f"  Trunc penalty: {scheduler.get_truncation_penalty(step):.2f}")

    # 保存历史
    scheduler.save_history("/tmp/temperature_history.csv")
    scheduler.plot_history("/tmp/temperature_history.png")

    print("\n" + "=" * 80)
    print("✅ Demo完成！")
    print("=" * 80)
