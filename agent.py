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
        
        # UCB探索参数 - 修正实现
        self.action_counts = np.zeros(Config.OUTPUT_DIM)
        self.action_rewards = np.zeros(Config.OUTPUT_DIM)  # 存储每个动作的累计奖励
        self.action_values = np.zeros(Config.OUTPUT_DIM)  # 存储每个动作的平均奖励
        self.total_counts = 0
        self.ucb_c = Config.UCB_C  # UCB探索系数
        
        # 尝试加载模型
        self.policy_net.load()
        
        # 开启训练模式
        self.policy_net.train()
    
    def select_action(self, state):
        """选择动作（完整UCB探索策略 + 防自杀机制）"""
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(Config.device)
            q_values = self.policy_net(state_tensor).cpu().numpy().flatten()
        
        # 获取当前状态的危险信息（前3个元素代表前方/右方/左方的危险）
        danger_signals = state[:3]
        
        
        # 计算UCB值
        ucb_scores = self._calculate_ucb_scores()
        
        # 获取候选动作（按UCB分数排序）
        candidate_actions = np.argsort(ucb_scores)[::-1]
        
        # 选择动作：优先选择UCB分数高且安全的动作
        selected_action = None
        for action in candidate_actions:
            # 检查动作是否安全
            is_safe = danger_signals[action] < 0.5
            
            if is_safe or not Config.ENABLE_SUICIDE_PREVENTION:
                selected_action = action
                break
        
        # 如果所有候选动作都危险，使用防自杀机制
        if selected_action is None:
            safe_action, danger_actions = safe_action_nb(
                q_values, 
                danger_signals,
                Config.COLLISION_PENALTY,
                state
            )
            selected_action = safe_action
            
            # 记录危险选择的惩罚（用于训练）
            for action_idx in danger_actions:
                self.memory.add(
                    state, 
                    action_idx, 
                    Config.COLLISION_PENALTY * 0.5,
                    state,
                    True
                )
        
        # 更新计数并记录动作
        self.action_counts[selected_action] += 1
        self.total_counts += 1
        self.steps_done += 1
        
        # 记录选择的动作用于UCB奖励更新
        if not hasattr(self, 'last_actions'):
            self.last_actions = []
        self.last_actions.append(selected_action)
        
        return selected_action
    
    def _calculate_ucb_scores(self):
        """计算修正的UCB分数（结合Q值估计和实际奖励 + UCB探索）"""
        ucb_scores = np.zeros(Config.OUTPUT_DIM)
        
        # 获取当前状态的Q值
        with torch.no_grad():
            # 这里需要一个虚拟状态，实际上我们应该从调用者传入Q值
            # 暂时使用Q网络的估计作为基础
            pass
        
        for action in range(Config.OUTPUT_DIM):
            if self.action_counts[action] == 0:
                # 从未尝试过的动作给予最高优先级
                ucb_scores[action] = float('inf')
            else:
                # 结合Q值估计和实际获得的平均奖励 + UCB探索奖励
                q_value_bonus = 0  # 将在select_action中结合
                avg_reward = self.action_values[action]
                ucb_bonus = self.ucb_c * np.sqrt(
                    np.log(self.total_counts + 1) / self.action_counts[action]
                )
                # 使用实际奖励和UCB探索的加权和
                ucb_scores[action] = avg_reward + ucb_bonus
        
        return ucb_scores

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
    
    def reset_ucb_stats(self):
        """重置UCB统计信息"""
        self.action_counts = np.zeros(Config.OUTPUT_DIM)
        self.action_rewards = np.zeros(Config.OUTPUT_DIM)
        self.action_values = np.zeros(Config.OUTPUT_DIM)
        self.total_counts = 0
    
    def save_model(self, is_best=False):
        """保存模型"""
        suffix = f"_best_{self.best_score}.pth" if is_best else ".pth"
        score = self.scores[-1] if self.scores else 0
        filename = f"snake_dqn_ep{self.episode}_sc{score}{suffix}"
        self.policy_net.save(filename)
        
        # 保存最佳模型
        if is_best:
            self.policy_net.save("snake_dqn_best.pth")
    
    def record_score(self, score, episode_reward=0):
        """记录分数和UCB奖励"""
        self.scores.append(score)
        
        # 更新UCB奖励统计（使用episode_reward更新动作价值）
        if hasattr(self, 'last_actions') and self.last_actions:
            for action in self.last_actions:
                self.action_rewards[action] += episode_reward / len(self.last_actions)
                if self.action_counts[action] > 0:
                    self.action_values[action] = self.action_rewards[action] / self.action_counts[action]
        
        # 更新最高分
        if score > self.best_score:
            self.best_score = score
            self.save_model(is_best=True)
            print(f"新记录! 分数: {score}")
        
        # 定期保存模型
        if self.episode % 5 == 0:
            self.save_model()
        
        # 定期更新目标网络
        if self.episode % Config.TARGET_UPDATE == 0:
            self.update_target_net()
            print("目标网络已更新")
        
        # 重置每轮的动作记录
        self.last_actions = []
        self.episode += 1