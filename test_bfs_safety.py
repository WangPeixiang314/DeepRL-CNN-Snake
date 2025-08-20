#!/usr/bin/env python3
"""
测试BFS不可达区域检测功能的脚本
"""

import numpy as np
from game import SnakeGame
from agent import DQNAgent
from config import Config
from help.numba_help.help_nb import check_unreachable_area_nb, enhanced_safe_action_nb

def test_bfs_safety():
    """测试BFS不可达区域检测功能"""
    print("=== 测试BFS不可达区域检测功能 ===")
    
    # 创建游戏实例
    game = SnakeGame(visualize=False)
    initial_state = game.get_state()
    input_dim = len(initial_state)
    agent = DQNAgent(input_dim)
    
    # 创建一个简单的场景来测试BFS检测
    game.reset()
    
    # 手动设置一个场景：蛇在一个角落，食物在另一个角落
    game.snake = [(0, 0), (0, 20), (0, 40)]  # 蛇在左上角
    game.head = (0, 0)
    game.food = (Config.WIDTH - 20, Config.HEIGHT - 20)  # 食物在右下角
    game.direction = game.direction.RIGHT  # 向右
    
    print(f"蛇头位置: {game.head}")
    print(f"食物位置: {game.food}")
    print(f"蛇身: {len(game.snake)}段")
    print(f"方向: {game.direction.value}")
    
    # 测试每个动作的不可达区域检测
    for action in [0, 1, 2]:  # 直行, 右转, 左转
        is_unreachable = check_unreachable_area_nb(
            game.snake,
            game.head,
            game.direction.value,
            game.food,
            game.width,
            game.height,
            Config.BLOCK_SIZE,
            action
        )
        
        action_names = ["直行", "右转", "左转"]
        print(f"动作 {action_names[action]} ({action}): 不可达 = {is_unreachable}")
    
    print("\n=== 测试增强版防自杀机制 ===")
    
    # 测试增强版防自杀机制
    state = game.get_state()
    danger_signals = state[:3]
    
    # 模拟Q值
    q_values = np.array([0.5, 0.3, 0.2])
    
    action, danger_actions = enhanced_safe_action_nb(
        q_values,
        danger_signals,
        Config.COLLISION_PENALTY,
        state,
        game.snake,
        game.head,
        game.direction.value,
        game.food,
        game.width,
        game.height,
        Config.BLOCK_SIZE
    )
    
    print(f"选择的动作: {action}")
    print(f"危险动作: {danger_actions}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_bfs_safety()