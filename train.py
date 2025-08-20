import logging
import numpy as np
import os
import pygame
import time
import torch
import re
from contextlib import contextmanager, redirect_stdout, redirect_stderr

from agent import DQNAgent
from config import Config
from game import SnakeGame
from training import TrainingPlotter, TrainingStats


@contextmanager
def suppress_output():
    # 保存原始日志级别
    original_log_level = logging.root.level
    logging.root.setLevel(logging.CRITICAL + 1)
    
    # 打开devnull
    with open(os.devnull, 'w') as devnull:
        # 重定向Python级别的stdout/stderr
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                yield
            finally:
                # 恢复日志级别
                logging.root.setLevel(original_log_level)

def train(num_episodes=1000, visualize=True, verbose=True, pretrained_model=None):
    # 检测GPU并设置设备
    Config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {Config.device}")
        if Config.device.type == 'cuda':
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # 定义内部训练函数
    def _train_internal():
        # 初始化组件
        game = SnakeGame(visualize=visualize)
        # 获取初始状态以确定输入维度
        initial_state = game.get_state()
        input_dim = len(initial_state)
        agent = DQNAgent(input_dim)
        
        # 如果有预训练模型，加载它
        if pretrained_model:
            model_path = os.path.join(Config.MODEL_DIR, pretrained_model)
            if os.path.exists(model_path):
                agent.policy_net.load(pretrained_model)
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                if verbose:
                    print(f"成功加载预训练模型: {pretrained_model}")
                
                # 尝试从文件名提取episode数
                try:
                    import re
                    match = re.search(r'ep(\d+)', pretrained_model)
                    if match:
                        start_episode = int(match.group(1))
                        agent.episode = start_episode
                        if verbose:
                            print(f"从第 {start_episode} 轮开始继续训练")
                except:
                    if verbose:
                        print("无法从文件名提取episode数，从第0轮开始")
            else:
                if verbose:
                    print(f"警告: 模型文件 {pretrained_model} 不存在，将从头开始训练")
        
        plotter = TrainingPlotter() if visualize else None
        stats = TrainingStats()
        
        # 训练变量
        start_time = time.time()
        
        # 定义训练循环函数
        def _run_training_loop():
            for _ in range(num_episodes):
                state = game.reset()
                done = False
                total_reward = 0
                episode_steps = 0
                episode_loss = []
                td_errors = []
                
                while not done:
                    # 选择动作
                    action = agent.select_action(state)
                    
                    # 执行动作
                    next_state, reward, done, score = game.step(action)
                    
                    # 如果退出事件被触发
                    if next_state is None:
                        if verbose:
                            print("检测到游戏退出信号，终止训练...")
                        break
                    
                    # 保存经验
                    agent.memory.add(state, action, reward, next_state, done)
                    
                    # 移动到下一个状态
                    state = next_state
                    total_reward += reward
                    episode_steps += 1
                    
                    # 优化模型
                    loss, td_error = agent.optimize_model()
                    if loss > 0:
                        episode_loss.append(loss)
                        if len(td_error) > 0:
                            td_errors.append(np.mean(td_error))
                
                # 计算平均损失
                avg_loss = sum(episode_loss) / len(episode_loss) if episode_loss else 0
                
                # 记录统计数据
                stats.update(score, total_reward, episode_steps, avg_loss, 
                            agent.epsilon_threshold, td_errors)
                
                # 记录结果
                agent.record_score(score, total_reward)
                
                # 定期更新图表
                if agent.episode % Config.PLOT_INTERVAL == 0:
                    if visualize:
                        plotter.update(stats.get_stats())
                        
                        # 计算并显示统计数据
                        elapsed_time = time.time() - start_time
                        time_per_episode = elapsed_time / agent.episode if agent.episode > 0 else 0
                        
                        # 打印当前TD误差统计
                        td_error_mean = np.mean(td_errors) if td_errors else 0
                        if verbose:
                            print(f"局数: {agent.episode}, 分数: {score}, 平均分数: {stats.avg_scores[-1] if stats.avg_scores else 0:.2f}, "
                                  f"损失: {avg_loss:.4f}, TD误差: {td_error_mean:.4f}, "
                                  f"时间/局: {time_per_episode:.2f}s, 经验池: {len(agent.memory)}, 探索率: {agent.epsilon_threshold:.4f}")
        
        # 执行训练循环
        _run_training_loop()
        
        if visualize and plotter is not None:
            plotter.close()
        return stats.avg_scores[-1] if stats.avg_scores else 0.0
    
    # 根据verbose决定是否抑制输出
    if verbose:
        return _train_internal()
    else:
        with suppress_output():
            return _train_internal()

# 程序入口
if __name__ == "__main__":
    import argparse
    import time
    
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='训练贪吃蛇AI')
    parser.add_argument('--model', type=str, help='预训练模型文件路径（相对于models目录）')
    parser.add_argument('--episodes', type=int, default=50000, help='训练轮次数量')
    parser.add_argument('--no-visualize', action='store_true', help='禁用可视化')
    parser.add_argument('--verbose', action='store_true', default=True, help='启用详细输出')
    
    args = parser.parse_args()
    
    begin = time.time()
    try:
        N = args.episodes
        
        # 如果有预训练模型，先加载
        if args.model:
            print(f"正在加载预训练模型: {args.model}")
            # 创建游戏实例获取状态维度
            game = SnakeGame(visualize=not args.no_visualize)
            initial_state = game.get_state()
            input_dim = len(initial_state)
            agent = DQNAgent(input_dim)
            
            # 加载指定模型
            model_path = os.path.join(Config.MODEL_DIR, args.model)
            if os.path.exists(model_path):
                agent.policy_net.load(args.model)
                agent.target_net.load_state_dict(agent.policy_net.state_dict())
                print(f"成功加载预训练模型: {args.model}")
                
                # 从已训练的episode数继续
                try:
                    # 从文件名中提取episode数
                    import re
                    match = re.search(r'ep(\d+)', args.model)
                    if match:
                        start_episode = int(match.group(1))
                        agent.episode = start_episode
                        print(f"从第 {start_episode} 轮开始继续训练")
                except:
                    print("无法从文件名提取episode数，从第0轮开始")
            else:
                print(f"警告: 模型文件 {args.model} 不存在，将从头开始训练")
        
        # 开始训练
        train(num_episodes=N, visualize=not args.no_visualize, verbose=args.verbose)
        
    except KeyboardInterrupt:
        print("训练已中断")
    finally:
        # 确保在退出前保存模型
        try:
            game = SnakeGame(visualize=False)  # 无可视化创建游戏实例
            initial_state = game.get_state()
            input_dim = len(initial_state)
            agent = DQNAgent(input_dim)
            agent.save_model()
            pygame.quit()
            print("模型已保存，程序退出")
            print(f"总训练耗时：{(time.time()-begin):.2f} s，总训练轮次：{N}。")
        except Exception as e:
            print(f"保存模型时出错: {e}")
            pygame.quit()