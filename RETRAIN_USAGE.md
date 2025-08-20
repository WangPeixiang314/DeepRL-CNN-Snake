# 再训练功能使用说明

## 功能介绍
本系统现在支持从已有的模型文件继续训练，让您可以在之前训练的基础上进一步提升模型性能。

## 使用方法

### 基本用法
```bash
python train.py --model 模型文件名
```

### 参数说明
- `--model`: 指定预训练模型文件路径（相对于models目录）
- `--episodes`: 设置训练轮次数量（默认：50000）
- `--no-visualize`: 禁用可视化界面
- `--verbose`: 启用详细输出（默认启用）

### 示例命令

1. 从特定模型继续训练：
```bash
python train.py --model snake_dqn_ep27005_sc54.pth
```

2. 从最佳模型继续训练：
```bash
python train.py --model snake_dqn_best.pth
```

3. 指定训练轮次：
```bash
python train.py --model snake_dqn_ep27005_sc54.pth --episodes 10000
```

4. 无可视化训练：
```bash
python train.py --model snake_dqn_ep27005_sc54.pth --no-visualize
```

## 注意事项

1. **模型文件位置**：请将预训练模型文件放在 `models/` 目录下
2. **episode计数**：系统会尝试从文件名中提取已训练的episode数，格式为 `ep{数字}`
3. **文件不存在**：如果指定的模型文件不存在，系统会从头开始训练
4. **兼容性**：确保预训练模型的网络结构与当前配置兼容

## 模型文件命名规则
模型文件通常采用以下格式：
- `snake_dqn_ep{episode数}_sc{分数}.pth` - 常规保存的模型
- `snake_dqn_best.pth` - 最佳性能模型

例如：`snake_dqn_ep27005_sc54.pth` 表示训练了27005轮，最高分数为54的模型。