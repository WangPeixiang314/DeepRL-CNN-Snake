import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Config

from help.numba_help.help_nb import safe_action_nb


from memory import PrioritizedReplayBuffer
from model import DQN


class DQNAgent:
    def __init__(self, input_dim):
        # 模型相关
        self.policy_net = DQN(input_dim, Config.HIDDEN_LAYERS, Config.OUTPUT_DIM)
        self.target_net = DQN(input_dim, Config.HIDDEN_LAYERS, Config.OUTPUT_DIM)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=Config.LEARNING_RATE)
        
        # 经验回放（使用优先级）
        self.memory = PrioritizedReplayBuffer(
            Config.MEMORY_CAPACITY,
            alpha=Config.PRIO_ALPHA,
            beta_start=Config.PRIO_BETA_START,
            beta_frames=Config.PRIO_BETA_FRAMES
        )
        
        # 训练参数
        self.steps_done = 0
        self.epsilon_threshold = Config.EPS_START
        self.episode = 0
        self.scores = []
        self.best_score = 0
        

        
        # 尝试加载模型
        self.policy_net.load()
        
        # 开启训练模式
        self.policy_net.train()
    
    def _calculate_epsilon(self, episode):
        """计算探索率的指数衰减策略（无周期性重启）
        
        Args:
            episode: 当前局数
            
        Returns:
            epsilon: 计算出的探索率
        """
        # 使用指数衰减替代线性衰减，更快的初期衰减
        max_episodes = Config.MAX_EPISODES
        
        # 指数衰减：epsilon = EPS_END + (EPS_START - EPS_END) * exp(-decay_rate * episode)
        # 设置衰减率使得在max_episodes时接近EPS_END
        decay_rate = 5.0 / max_episodes  # 调整这个值可以改变衰减速度
        
        epsilon = Config.EPS_END + (Config.EPS_START - Config.EPS_END) * np.exp(-decay_rate * episode)
        
        # 确保epsilon不小于最小值
        return max(Config.EPS_END, epsilon)

    def select_action(self, state):
        """选择动作（使用指数衰减ε-贪婪策略）"""
        # 使用指数衰减计算探索率（无周期性重启）
        self.epsilon_threshold = self._calculate_epsilon(self.episode)
        
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(Config.device)
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        
        # 获取当前状态的危险信息（前3个元素代表前方/右方/左方的危险）
        danger_signals = state[:3]
        
        # ε-贪婪策略
        if np.random.random() < self.epsilon_threshold:
            # 探索：随机选择安全动作
            safe_actions = [i for i, danger in enumerate(danger_signals) if danger < 0.5]
            if safe_actions:
                action = np.random.choice(safe_actions)
            else:
                # 如果没有安全动作，使用防自杀机制
                safe_action, _ = safe_action_nb(
                    q_values, 
                    danger_signals,
                    Config.COLLISION_PENALTY,
                    state
                )
                action = safe_action
        else:
            # 利用：选择Q值最高的安全动作
            safe_q_values = q_values.copy()
            for i, danger in enumerate(danger_signals):
                if danger >= 0.5 and Config.ENABLE_SUICIDE_PREVENTION:
                    safe_q_values[i] = float('-inf')
            
            action = np.argmax(safe_q_values)
            
            # 如果所有动作都危险，使用防自杀机制
            if safe_q_values[action] == float('-inf'):
                safe_action, _ = safe_action_nb(
                    q_values, 
                    danger_signals,
                    Config.COLLISION_PENALTY,
                    state
                )
                action = safe_action
        
        self.steps_done += 1
        return action
    
    def optimize_model(self):
        """优化模型（采用优先级采样）"""
        if len(self.memory) < Config.BATCH_SIZE:
            return 0.0, []
        
        # 从内存中采样批次（带权重）
        (states, actions, rewards, next_states, dones, 
        indices, weights) = self.memory.sample(Config.BATCH_SIZE)
        
        # 修复警告：使用 detach().clone() 替代 torch.tensor()
        weights = weights.detach().clone().to(Config.device)
        
        # 计算当前状态的Q值
        state_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # 计算下一个状态的最大Q值 (Double DQN)
        with torch.no_grad():
            # 使用策略网络选择最优动作
            next_actions = self.policy_net(next_states).max(1)[1]
            # 使用目标网络评估这些动作的Q值
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        
        # 计算期望Q值 (Bellman方程)
        expected_q_values = rewards + (1 - dones) * Config.GAMMA * next_q_values
        
        # 计算损失（加入重要性采样权重）
        loss = F.smooth_l1_loss(state_q_values.squeeze(), expected_q_values, reduction='none')
        weighted_loss = (weights * loss).mean()
        
        # 计算TD误差（用于优先级）
        td_errors = loss.detach().cpu().numpy()
        
        # 优化模型
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # 梯度裁剪
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # 更新优先级
        self.memory.update_priorities(indices, td_errors)
        
        return weighted_loss.item(), td_errors

    def update_target_net(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, is_best=False):
        """保存模型"""
        suffix = f"_best_{self.best_score}.pth" if is_best else ".pth"
        score = self.scores[-1] if self.scores else 0
        filename = f"snake_dqn_ep{self.episode}_sc{score}{suffix}"
        self.policy_net.save(filename)
        
        # 保存最佳模型
        if is_best:
            self.policy_net.save("snake_dqn_best.pth")
    
    def load_model(self, filename):
        """加载指定模型文件"""
        return self.policy_net.load(filename)
    
    def record_score(self, score, episode_reward=0):
        """记录分数"""
        self.scores.append(score)
        
        # 更新最高分
        if score > self.best_score:
            self.best_score = score
            self.save_model(is_best=True)
            print(f"新记录! 分数: {score}")
        
        # 定期保存模型
        if self.episode % 100 == 0:
            self.save_model()
        
        # 定期更新目标网络
        if self.episode % Config.TARGET_UPDATE == 0:
            self.update_target_net()
            print("目标网络已更新")
        
        # 重置每轮的动作记录
        if hasattr(self, 'last_actions'):
            self.last_actions = []
        self.episode += 1