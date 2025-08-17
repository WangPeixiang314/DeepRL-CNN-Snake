import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config


class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器 - 用于处理游戏地图的卷积神经网络"""
    
    def __init__(self, input_channels=None, feature_dim=None):
        super(CNNFeatureExtractor, self).__init__()
        
        # 使用配置参数
        config = Config.CNN_ALL_GRID_CONFIG
        if input_channels is None:
            input_channels = config['input_channels']
        if feature_dim is None:
            feature_dim = config['feature_dim']
        
        # 动态创建CNN层
        conv_layers = config['conv_layers']
        self.conv_layers = nn.ModuleList()
        
        in_channels = input_channels
        for i, out_channels in enumerate(conv_layers):
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=config['kernel_size'], 
                         stride=config['stride'], 
                         padding=config['padding'])
            )
            in_channels = out_channels
        
        # 池化层配置
        self.use_pooling = config.get('use_pooling', True)
        self.pool_layers = config.get('pool_layers', 2)
        if self.use_pooling:
            self.pool = nn.MaxPool2d(config['pool_size'], config['pool_size'])
        
        # 自适应池化层
        self.adaptive_pool = nn.AdaptiveAvgPool2d(config['adaptive_pool_size'])
        
        # 动态计算全连接层输入维度
        fc_input_dim = conv_layers[-1] * config['adaptive_pool_size'][0] * config['adaptive_pool_size'][1]
        self.fc = nn.Linear(fc_input_dim, feature_dim)
        
    def forward(self, x):
        """前向传播"""
        for i, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            # 根据配置使用池化
            if self.use_pooling and i < self.pool_layers and i < len(self.conv_layers) - 1:
                x = self.pool(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc(x))
        return x

class LocalCNNFeatureExtractor(nn.Module):
    """局部视野CNN特征提取器"""
    
    def __init__(self, input_channels=None, feature_dim=None):
        super(LocalCNNFeatureExtractor, self).__init__()
        
        # 使用配置参数
        config = Config.CNN_LOCAL_CONFIG
        if input_channels is None:
            input_channels = config['input_channels']
        if feature_dim is None:
            feature_dim = config['feature_dim']
        
        # 动态创建CNN层
        conv_layers = config['conv_layers']
        self.conv_layers = nn.ModuleList()
        
        in_channels = input_channels
        for out_channels in conv_layers:
            self.conv_layers.append(
                nn.Conv2d(in_channels, out_channels, 
                         kernel_size=config['kernel_size'], 
                         stride=config['stride'], 
                         padding=config['padding'])
            )
            in_channels = out_channels
        
        # 自适应池化到固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d(config['adaptive_pool_size'])
        
        # 动态计算全连接层输入维度
        fc_input_dim = conv_layers[-1] * config['adaptive_pool_size'][0] * config['adaptive_pool_size'][1]
        self.fc = nn.Linear(fc_input_dim, feature_dim)
        
    def forward(self, x):
        """前向传播"""
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)  # 展平
        x = F.relu(self.fc(x))
        return x

# 创建全局CNN特征提取器实例
all_grid_cnn = CNNFeatureExtractor().to(Config.device)
local_cnn = LocalCNNFeatureExtractor().to(Config.device)

def get_CNN_all_grid_view(game_state):
    """
    对整个地图执行CNN处理并返回展平的一维向量
    
    参数:
        game_state: 包含游戏状态的对象
        
    返回:
        np.ndarray: CNN处理后的特征向量
    """
    # 创建3通道的地图表示
    # 通道0: 蛇身位置 (1表示蛇身，0表示空)
    # 通道1: 食物位置 (1表示食物，0表示空)  
    # 通道2: 蛇头位置 (1表示蛇头，0表示空)
    
    grid_width = Config.GRID_WIDTH
    grid_height = Config.GRID_HEIGHT
    
    # 初始化3通道地图
    full_map = np.zeros((3, grid_height, grid_width), dtype=np.float32)
    
    # 填充蛇身信息
    for segment in game_state.snake:
        x, y = int(segment[0] // Config.BLOCK_SIZE), int(segment[1] // Config.BLOCK_SIZE)
        if 0 <= x < grid_width and 0 <= y < grid_height:
            full_map[0, y, x] = 1.0
    
    # 填充食物信息
    food_x, food_y = int(game_state.food[0] // Config.BLOCK_SIZE), int(game_state.food[1] // Config.BLOCK_SIZE)
    if 0 <= food_x < grid_width and 0 <= food_y < grid_height:
        full_map[1, food_y, food_x] = 1.0
    
    # 填充蛇头信息
    head_x, head_y = int(game_state.head[0] // Config.BLOCK_SIZE), int(game_state.head[1] // Config.BLOCK_SIZE)
    if 0 <= head_x < grid_width and 0 <= head_y < grid_height:
        full_map[2, head_y, head_x] = 1.0
    
    # 转换为PyTorch张量
    map_tensor = torch.from_numpy(full_map).unsqueeze(0).to(Config.device)
    
    # 使用CNN提取特征
    with torch.no_grad():
        features = all_grid_cnn(map_tensor)
    
    # 返回numpy数组
    return features.cpu().numpy().flatten()

def get_CNN_local_grid_view(game_state):
    """
    对局部视野执行CNN处理并返回展平的一维向量
    
    参数:
        game_state: 包含游戏状态的对象
        
    返回:
        np.ndarray: CNN处理后的局部特征向量
    """
    # 获取蛇头位置
    head_x = int(game_state.head[0] // Config.BLOCK_SIZE)
    head_y = int(game_state.head[1] // Config.BLOCK_SIZE)
    
    # 定义6x6的局部视野
    local_radius = 3
    local_map = np.zeros((3, 6, 6), dtype=np.float32)
    
    # 填充局部地图信息
    for dx in range(-local_radius, local_radius):
        for dy in range(-local_radius, local_radius):
            grid_x = head_x + dx
            grid_y = head_y + dy
            
            # 检查是否在边界内
            if (0 <= grid_x < Config.GRID_WIDTH and 
                0 <= grid_y < Config.GRID_HEIGHT):
                
                # 转换为局部坐标
                local_x = dx + local_radius
                local_y = dy + local_radius
                
                # 检查该位置的内容
                pixel_x = grid_x * Config.BLOCK_SIZE
                pixel_y = grid_y * Config.BLOCK_SIZE
                
                # 蛇身
                if (pixel_x, pixel_y) in game_state.snake:
                    if (pixel_x, pixel_y) == game_state.head:
                        local_map[2, local_y, local_x] = 1.0  # 蛇头
                    else:
                        local_map[0, local_y, local_x] = 1.0  # 蛇身
                
                # 食物
                if (pixel_x, pixel_y) == game_state.food:
                    local_map[1, local_y, local_x] = 1.0  # 食物
    
    # 转换为PyTorch张量
    local_tensor = torch.from_numpy(local_map).unsqueeze(0).to(Config.device)
    
    # 使用局部CNN提取特征
    with torch.no_grad():
        features = local_cnn(local_tensor)
    
    # 返回numpy数组
    return features.cpu().numpy().flatten()



# 动态计算的CNN特征维度
CNN_ALL_GRID_FEATURES = Config.CNN_ALL_GRID_OUTPUT_DIM  # 全局CNN特征维度
CNN_LOCAL_FEATURES = Config.CNN_LOCAL_OUTPUT_DIM       # 局部CNN特征维度
ALL_GRID_FLAT_FEATURES = Config.GRID_WIDTH * Config.GRID_HEIGHT  # 全地图展平特征维度