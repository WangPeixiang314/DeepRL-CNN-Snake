import os
import os
import torch
import torch


class Config:
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 网格参数
    GRID_WIDTH = 12   # 地图网格宽度（单位：格）
    GRID_HEIGHT = 12  # 地图网格高度（单位：格）
    BLOCK_SIZE = 40   # 每个网格块的像素大小
    WIDTH = GRID_WIDTH * BLOCK_SIZE
    HEIGHT = GRID_HEIGHT * BLOCK_SIZE
    
    # 游戏参数
    SPEED = 600
    
    # 模型参数
    HIDDEN_LAYERS = [256, 128, 64, 32]  # 隐藏层配置
    OUTPUT_DIM = 3   # [直行, 右转, 左转]
    
    # CNN网络参数 - 全局地图CNN
    CNN_ALL_GRID_OUTPUT_DIM = 64  # 全局CNN输出特征维度
    CNN_ALL_GRID_CONFIG = {
        'input_channels': 3,    # 输入通道数 [蛇身, 食物, 蛇头]
        'conv_layers': [16, 32, 64],  # 每层卷积核数量
        'kernel_size': 3,       # 卷积核大小
        'stride': 1,            # 步长
        'padding': 1,           # 填充
        'pool_size': 2,         # 池化大小 (仅在前两层使用)
        'feature_dim': CNN_ALL_GRID_OUTPUT_DIM,  # 输出特征维度
        'adaptive_pool_size': (3, 3)  # 自适应池化输出大小 (增大避免信息损失)
    }
    
    # CNN网络参数 - 局部视野CNN
    CNN_LOCAL_OUTPUT_DIM = 32  # 局部CNN输出特征维度
    CNN_LOCAL_CONFIG = {
        'input_channels': 3,    # 输入通道数 [蛇身, 食物, 蛇头]
        'conv_layers': [8, 16], # 每层卷积核数量
        'kernel_size': 3,       # 卷积核大小
        'stride': 1,            # 步长
        'padding': 1,           # 填充
        'feature_dim': CNN_LOCAL_OUTPUT_DIM,  # 输出特征维度
        'adaptive_pool_size': (2, 2)  # 自适应池化输出大小
    }
    
    # 训练参数
    BATCH_SIZE = 128
    MEMORY_CAPACITY = 800_000
    LEARNING_RATE = 3.482e-05
    GAMMA = 0.99
    TARGET_UPDATE = 100  # 更新目标网络的间隔
    MAX_STEPS_WITHOUT_FOOD = 500  # 最大无食物步数
    
    # 优先级采样参数
    PRIO_ALPHA = 0.6  # 控制采样的随机性程度 (0~1)
    PRIO_BETA_START = 0.4  # 重要性采样权重的初始值
    PRIO_BETA_FRAMES = 60_000  # beta增加到1所需的帧数
    PRIO_EPS = 1e-6  # 防止优先级为0的小常数
    
    # 探索参数
    EPS_START = 1.0
    EPS_END = 0.02
    EPS_DECAY = 50000  # ε衰减步数 (增加以延长探索)
    
    # 奖励参数
    FOOD_REWARD = 16.0
    COLLISION_PENALTY = -10.0
    STEP_PENALTY = 0.01
    PROGRESS_REWARD = 0.1  # 向食物靠近的奖励
    
    # 文件路径
    MODEL_DIR = './models'
    MODEL_FILE = 'snake_dqn.pth'
    LOG_FILE = 'training_log.csv'
    
    # 绘图参数
    PLOT_INTERVAL = 2  # 每N局游戏更新绘图 
    
    @staticmethod
    def init():
        """创建必要的目录"""
        if not os.path.exists(Config.MODEL_DIR):
            os.makedirs(Config.MODEL_DIR)


# 初始化配置
Config.init()