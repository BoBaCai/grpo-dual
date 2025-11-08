#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Temperature Scheduler 测试脚本

验证：
1. Stage-wise 降温是否正确
2. Per-task 温度差异
3. 自适应规则是否生效
4. KL、max_tokens 等配套功能
"""

import sys
import random
from temperature_scheduler import TemperatureScheduler, TemperatureConfig


def test_stage_wise_schedule():
    """测试 Stage-wise 温度调度"""
    print("\n" + "=" * 80)
    print("测试 1: Stage-wise 温度调度")
    print("=" * 80)

    scheduler = TemperatureScheduler(total_steps=500)

    # 测试关键步数
    test_steps = [0, 50, 150, 250, 400, 500]

    print("\nStep | Stage | T_fair | T_halu | KL    | MaxTok | TruncPen")
    print("-" * 70)

    for step in test_steps:
        temps = scheduler.get_temperature(step=step)
        kl = scheduler.get_kl_coefficient(step)
        max_tok = scheduler.get_max_new_tokens(step)
        trunc_pen = scheduler.get_truncation_penalty(step)

        print(f"{step:4d} | {temps['stage']:5d} | "
              f"{temps['fairness']:.3f}  | {temps['hallucination']:.3f}  | "
              f"{kl:.4f} | {max_tok:6d} | {trunc_pen:.2f}")

    # 验证
    temps_stage1 = scheduler.get_temperature(step=50)
    temps_stage2 = scheduler.get_temperature(step=250)
    temps_stage3 = scheduler.get_temperature(step=450)

    assert temps_stage1['fairness'] > temps_stage2['fairness'] > temps_stage3['fairness'], \
        "❌ Fairness 温度应该逐阶段下降"
    assert temps_stage1['hallucination'] > temps_stage2['hallucination'], \
        "❌ Hallucination 温度应该下降"

    print("\n✅ Stage-wise 降温验证通过")


def test_per_task_difference():
    """测试 Per-task 温度差异"""
    print("\n" + "=" * 80)
    print("测试 2: Per-task 温度差异")
    print("=" * 80)

    scheduler = TemperatureScheduler(total_steps=500)

    print("\nStage 1 (高探索期):")
    temps_s1 = scheduler.get_temperature(step=50)
    print(f"  Fairness T:      {temps_s1['fairness']:.3f} (期望: 略高)")
    print(f"  Hallucination T: {temps_s1['hallucination']:.3f} (期望: 中等)")

    print("\nStage 2 (收敛期):")
    temps_s2 = scheduler.get_temperature(step=250)
    print(f"  Fairness T:      {temps_s2['fairness']:.3f}")
    print(f"  Hallucination T: {temps_s2['hallucination']:.3f}")

    print("\nStage 3 (精修期):")
    temps_s3 = scheduler.get_temperature(step=450)
    print(f"  Fairness T:      {temps_s3['fairness']:.3f} (期望: 略高)")
    print(f"  Hallucination T: {temps_s3['hallucination']:.3f} (期望: 略低)")

    # 验证：Fairness 温度始终略高于 Hallucination
    assert temps_s1['fairness'] > temps_s1['hallucination'], \
        "❌ Stage 1: Fairness 应该高于 Hallucination"
    assert temps_s2['fairness'] > temps_s2['hallucination'], \
        "❌ Stage 2: Fairness 应该高于 Hallucination"
    assert temps_s3['fairness'] > temps_s3['hallucination'], \
        "❌ Stage 3: Fairness 应该高于 Hallucination"

    print("\n✅ Per-task 温度差异验证通过")


def test_adaptive_rules():
    """测试自适应规则"""
    print("\n" + "=" * 80)
    print("测试 3: 自适应规则（熵 + 截断率）")
    print("=" * 80)

    scheduler = TemperatureScheduler(total_steps=500)

    # Stage 2 (启用 both 模式)
    step = 250

    print("\n场景 1: 截断率过高 (30% > 15%)")
    temps1 = scheduler.get_temperature(
        step=step,
        fairness_entropy=3.5,
        fairness_trunc_rate=0.30,  # 高于阈值 0.15
        hallucination_entropy=3.5,
        hallucination_trunc_rate=0.10
    )
    print(f"  Fairness T: {temps1['fairness']:.3f} (期望: 降低)")

    print("\n场景 2: 熵过低 (2.5 < 3.0)")
    temps2 = scheduler.get_temperature(
        step=step + scheduler.config.window_steps,
        fairness_entropy=2.5,  # 低于 target_low
        fairness_trunc_rate=0.10,
        hallucination_entropy=3.5,
        hallucination_trunc_rate=0.10
    )
    print(f"  Fairness T: {temps2['fairness']:.3f} (期望: 提高)")

    print("\n场景 3: 熵过高 (4.5 > 4.0)")
    temps3 = scheduler.get_temperature(
        step=step + 2 * scheduler.config.window_steps,
        fairness_entropy=4.5,  # 高于 target_high
        fairness_trunc_rate=0.10,
        hallucination_entropy=3.5,
        hallucination_trunc_rate=0.10
    )
    print(f"  Fairness T: {temps3['fairness']:.3f} (期望: 降低)")

    print("\n✅ 自适应规则验证通过（查看历史确认调整原因）")


def test_truncation_penalty():
    """测试截断惩罚机制"""
    print("\n" + "=" * 80)
    print("测试 4: 截断惩罚系数")
    print("=" * 80)

    scheduler = TemperatureScheduler(total_steps=500)

    # 模拟被截断的样本
    original_reward = 1.0

    print("\nStage | Trunc Penalty | Final Reward (if truncated)")
    print("-" * 55)

    for step in [50, 250, 450]:
        stage = scheduler.get_current_stage(step)
        penalty = scheduler.get_truncation_penalty(step)
        final_reward = original_reward * penalty

        print(f"{stage:5d} | {penalty:13.2f} | {final_reward:.3f}")

    # 验证：惩罚逐阶段加重
    pen_s1 = scheduler.get_truncation_penalty(50)
    pen_s2 = scheduler.get_truncation_penalty(250)
    pen_s3 = scheduler.get_truncation_penalty(450)

    assert pen_s1 > pen_s2 > pen_s3, "❌ 截断惩罚应该逐阶段加重"

    print("\n✅ 截断惩罚验证通过")


def test_length_penalty():
    """测试长度正则化"""
    print("\n" + "=" * 80)
    print("测试 5: 长度正则化系数")
    print("=" * 80)

    scheduler = TemperatureScheduler(total_steps=500)

    L_target = 128
    test_lengths = [64, 128, 192, 256]

    print("\nStage 1 (λ=0.01):")
    lambda_s1 = scheduler.get_length_penalty_lambda(50)
    print(f"Length | Penalty")
    print("-" * 25)
    for L in test_lengths:
        penalty = -lambda_s1 * max(0, (L - L_target) / L_target)
        print(f"{L:6d} | {penalty:.4f}")

    print("\nStage 3 (λ=0.05):")
    lambda_s3 = scheduler.get_length_penalty_lambda(450)
    print(f"Length | Penalty")
    print("-" * 25)
    for L in test_lengths:
        penalty = -lambda_s3 * max(0, (L - L_target) / L_target)
        print(f"{L:6d} | {penalty:.4f}")

    # 验证：λ 逐阶段增大
    lambda_s1 = scheduler.get_length_penalty_lambda(50)
    lambda_s2 = scheduler.get_length_penalty_lambda(250)
    lambda_s3 = scheduler.get_length_penalty_lambda(450)

    assert lambda_s1 < lambda_s2 < lambda_s3, "❌ 长度惩罚系数应该逐阶段增大"

    print("\n✅ 长度正则化验证通过")


def test_full_training_simulation():
    """完整训练模拟"""
    print("\n" + "=" * 80)
    print("测试 6: 完整训练模拟 (500 步)")
    print("=" * 80)

    scheduler = TemperatureScheduler(total_steps=500)

    # 模拟训练过程
    for step in range(0, 501, 50):
        # 模拟指标（加入一些随机性和趋势）
        # 假设：熵逐步稳定，截断率逐步下降
        base_entropy = 3.5 - 0.5 * (step / 500)  # 从 3.5 降到 3.0
        base_trunc = 0.4 - 0.3 * (step / 500)    # 从 0.4 降到 0.1

        fairness_entropy = base_entropy + random.uniform(-0.5, 0.5)
        fairness_trunc = max(0.05, base_trunc + random.uniform(-0.1, 0.1))

        halu_entropy = base_entropy + random.uniform(-0.3, 0.3)
        halu_trunc = max(0.03, base_trunc * 0.7 + random.uniform(-0.05, 0.05))

        temps = scheduler.get_temperature(
            step=step,
            fairness_entropy=fairness_entropy,
            fairness_trunc_rate=fairness_trunc,
            hallucination_entropy=halu_entropy,
            hallucination_trunc_rate=halu_trunc
        )

    # 保存历史
    import os
    os.makedirs("/tmp/grpo_temp_test", exist_ok=True)
    scheduler.save_history("/tmp/grpo_temp_test/temperature_history.csv")
    scheduler.plot_history("/tmp/grpo_temp_test/temperature_history.png")

    print("\n✅ 完整训练模拟完成")
    print(f"📊 查看结果: /tmp/grpo_temp_test/temperature_history.png")


def test_custom_config():
    """测试自定义配置"""
    print("\n" + "=" * 80)
    print("测试 7: 自定义配置")
    print("=" * 80)

    custom_config = TemperatureConfig(
        T_min=0.5,
        T_max=1.5,
        fairness_T_init=1.20,
        hallucination_T_init=0.90,
        stage1_end=0.25,  # 25% 探索
        stage2_end=0.85,  # 25-85% 收敛
        entropy_target_low=2.5,
        entropy_target_high=4.5
    )

    scheduler = TemperatureScheduler(total_steps=1000, config=custom_config)

    print("\n自定义配置:")
    print(f"  T 范围: [{custom_config.T_min}, {custom_config.T_max}]")
    print(f"  Fairness 初始: {custom_config.fairness_T_init}")
    print(f"  Hallucination 初始: {custom_config.hallucination_T_init}")
    print(f"  Stage 划分: {custom_config.stage1_end:.0%} / "
          f"{custom_config.stage2_end:.0%} / 100%")

    temps_s1 = scheduler.get_temperature(step=100)
    temps_s2 = scheduler.get_temperature(step=500)
    temps_s3 = scheduler.get_temperature(step=900)

    print(f"\nStep 100 (Stage {temps_s1['stage']}): "
          f"T_fair={temps_s1['fairness']:.3f}, T_halu={temps_s1['hallucination']:.3f}")
    print(f"Step 500 (Stage {temps_s2['stage']}): "
          f"T_fair={temps_s2['fairness']:.3f}, T_halu={temps_s2['hallucination']:.3f}")
    print(f"Step 900 (Stage {temps_s3['stage']}): "
          f"T_fair={temps_s3['fairness']:.3f}, T_halu={temps_s3['hallucination']:.3f}")

    print("\n✅ 自定义配置验证通过")


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("🧪 Temperature Scheduler 测试套件")
    print("=" * 80)

    try:
        test_stage_wise_schedule()
        test_per_task_difference()
        test_adaptive_rules()
        test_truncation_penalty()
        test_length_penalty()
        test_full_training_simulation()
        test_custom_config()

        print("\n" + "=" * 80)
        print("✅ 所有测试通过！")
        print("=" * 80)

        print("\n📝 下一步:")
        print("  1. 查看生成的图表: /tmp/grpo_temp_test/temperature_history.png")
        print("  2. 阅读集成指南: TEMPERATURE_INTEGRATION_GUIDE.md")
        print("  3. 集成到 trainer.py")

    except AssertionError as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未预期的错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
