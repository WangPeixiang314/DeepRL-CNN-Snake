# Numba加速配置说明

## 功能概述

现在支持通过配置文件动态切换Numba加速功能，无需修改代码即可在Numba加速和Python原生实现之间切换。

## 配置方法

### 1. 修改配置文件

在 `config.py` 中找到以下配置项：

```python
# 性能优化配置
USE_NUMBA = True  # 设置为True启用Numba加速，False使用Python原生版本
```

### 2. 支持的模块

以下模块支持Numba配置切换：

- **game.py**: 游戏核心逻辑
- **agent.py**: 智能体决策逻辑  
- **memory.py**: 经验回放和SumTree实现

### 3. 使用示例

#### 启用Numba加速（默认）
```python
# config.py
USE_NUMBA = True
```

#### 禁用Numba加速
```python
# config.py  
USE_NUMBA = False
```

### 4. 运行时切换

也可以在运行时动态修改配置：

```python
from config import Config

# 禁用Numba加速
Config.USE_NUMBA = False

# 重新导入相关模块（需要重启程序）
```

## 性能对比

| 配置 | 启动速度 | 运行速度 | 内存占用 |
|------|----------|----------|----------|
| Numba启用 | 较慢 | 极快 | 较高 |
| Numba禁用 | 快速 | 一般 | 较低 |

## 注意事项

1. **首次启用Numba**会有编译开销，后续运行会显著提速
2. **切换配置后需要重启程序**才能生效
3. **Numba加速适合长时间训练**，短期测试可能看不出明显优势
4. **Python原生版本兼容性更好**，适合调试和开发阶段

## 故障排除

如果切换配置后出现问题：

1. 检查 `help/Python_help/help_py.py` 中的函数是否与Numba版本兼容
2. 确保所有必要的Python原生函数都已实现
3. 重启Python解释器确保配置生效