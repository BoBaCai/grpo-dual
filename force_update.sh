#!/bin/bash
#
# 强制更新并验证脚本
# 解决 temperature 警告 + KL 爆炸问题
#

set -e

echo "=========================================="
echo "GRPO 代码更新与验证"
echo "=========================================="

# 1. 保存当前修改（如果有）
echo -e "\n[1/5] 检查本地修改..."
if ! git diff --quiet; then
    echo "⚠️ 检测到本地修改，正在暂存..."
    git stash
    echo "✅ 已暂存（稍后可用 git stash pop 恢复）"
else
    echo "✅ 无本地修改"
fi

# 2. 拉取最新代码
echo -e "\n[2/5] 拉取最新代码..."
git pull origin claude/update-sandbox-file-011CUQrYCAKa4Jz2jD4YMVEn
echo "✅ 代码已更新"

# 3. 验证关键修复
echo -e "\n[3/5] 验证关键修复..."

# 检查 temperature 修复
if grep -q "TemperatureLogitsWarper(config.TEMPERATURE_TRAIN)" grpo-dual/scripts/grpo_train.py; then
    echo "✅ Temperature 修复已存在"
else
    echo "❌ Temperature 修复缺失！"
    exit 1
fi

# 检查 KL 修复
if grep -q "delta = (ref_lp - cur_lp).clamp(-10, 10)" grpo-dual/scripts/grpo_train.py; then
    echo "✅ KL 散度修复已存在"
else
    echo "❌ KL 散度修复缺失！"
    exit 1
fi

# 检查梯度累积
if grep -q "GRADIENT_ACCUMULATION_STEPS = 2" grpo-dual/scripts/grpo_train.py; then
    echo "✅ 梯度累积配置已存在"
else
    echo "❌ 梯度累积配置缺失！"
    exit 1
fi

# 4. 显示当前配置
echo -e "\n[4/5] 当前配置:"
echo "----------------------------------------"
grep "GRPO_BATCH_SIZE\|GRADIENT_ACCUMULATION_STEPS\|MAX_NEW_TOKENS_TRAIN\|LORA_R\|TEMPERATURE_TRAIN" grpo-dual/scripts/grpo_train.py | head -10
echo "----------------------------------------"

# 5. 重启提示
echo -e "\n[5/5] ⚠️ 重要提示："
echo "如果你在 Jupyter Notebook 中运行，请："
echo "   1. 停止当前运行（如果有）"
echo "   2. 点击 Kernel → Restart Kernel"
echo "   3. 重新运行所有 cell"
echo ""
echo "或者重新运行训练脚本:"
echo "   python grpo-dual/scripts/grpo_train.py"
echo ""
echo "=========================================="
echo "✅ 更新完成！"
echo "=========================================="
