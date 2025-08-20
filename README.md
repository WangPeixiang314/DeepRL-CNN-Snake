# 深度强化学习贪吃蛇 AI

本项目使用深度Q网络(DQN)训练贪吃蛇AI，支持从预训练模型继续训练。

## 功能特性

- 🧠 **深度强化学习**：使用DQN算法训练智能贪吃蛇
- 🔄 **再训练支持**：可以从已有模型继续训练
- 📊 **实时监控**：可视化训练过程和性能指标
- ⚡ **高性能**：支持GPU加速训练
- 🎯 **优先级经验回放**：提高训练效率

## 快速开始

### 基础训练
```bash
# 从头开始训练
python train.py

# 训练指定轮次
python train.py --episodes 10000
```

### 再训练功能
```bash
# 从预训练模型继续训练
python train.py --model snake_dqn_ep27005_sc54.pth

# 从最佳模型继续训练
python train.py --model snake_dqn_best.pth

# 指定训练轮次
python train.py --model snake_dqn_ep27005_sc54.pth --episodes 5000

# 无可视化训练
python train.py --model snake_dqn_best.pth --no-visualize
```

### 命令行参数
- `--model MODEL`: 预训练模型文件路径（相对于models目录）
- `--episodes EPISODES`: 训练轮次数量（默认：50000）
- `--no-visualize`: 禁用可视化界面
- `--verbose`: 启用详细输出

## 项目结构

```
DeepRL-CNN-Snake/
├── train.py              # 主训练脚本
├── agent.py              # DQN智能体实现
├── model.py              # 神经网络模型
├── game.py               # 贪吃蛇游戏环境
├── config.py             # 配置参数
├── models/               # 模型文件目录
│   ├── snake_dqn_best.pth
│   └── snake_dqn_ep27005_sc54.pth
├── RETRAIN_USAGE.md     # 再训练功能详细说明
└── test_retrain.py      # 再训练功能测试脚本
```

## 性能优化

- **GPU加速**：自动检测并使用可用GPU
- **优先级经验回放**：提高样本利用效率
- **指数衰减**：优化探索策略
- **防自杀机制**：避免AI自我毁灭

## 注意事项

1. **模型兼容性**：确保预训练模型与当前网络结构兼容
2. **文件路径**：模型文件需放在`models/`目录下
3. **episode计数**：系统会自动从文件名提取已训练轮次
4. **内存需求**：建议至少8GB RAM用于训练

## 故障排除

如果遇到问题，请运行测试脚本：
```bash
python test_retrain.py
```

## 许可证

MIT License - 详见LICENSE文件