import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

optuna_logger = logging.getLogger('optuna')
optuna_logger.setLevel(logging.ERROR)

import optuna
import multiprocessing as mp
from config import Config
from train import train

def train_wrapper(args):
    """训练包装函数，用于多进程调用"""
    num_episodes, visualize, verbose = args
    return train(num_episodes=num_episodes, visualize=visualize, verbose=verbose)

def objective(trial):
    # 1. 学习率
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    
    # 2. 批量大小
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    
    # 3. CNN维度
    cnn_all_dim = trial.suggest_categorical('cnn_all_dim', [32, 64, 128, 256])
    cnn_local_dim = trial.suggest_categorical('cnn_local_dim', [16, 32, 64, 128])
    
    # 4. 网络架构
    layer_count = trial.suggest_int('layer_count', 2, 5)
    base_size = trial.suggest_categorical('base_size', [128, 256, 512, 1024])
    size_decay = trial.suggest_float('size_decay', 0.5, 0.9, step=0.1)


    # 5. 训练参数
    gamma = trial.suggest_float('gamma', 0.95, 0.999, step=0.001)

    target_update = trial.suggest_int('target_update', 50, 500, step=50)

    # 6. 奖励参数
    food_reward = trial.suggest_float('food_reward', 10.0, 30.0, step=1.0)
    death_penalty = trial.suggest_float('death_penalty', -30.0, -10.0, step=1.0)
    progress_reward = trial.suggest_float('progress_reward', 0.0, 5.0, step=0.5)
    step_penalty = trial.suggest_float('step_penalty', -0.5, 0.5, step=0.1)
    
    # 7. 探索参数
    eps_decay = trial.suggest_int('eps_decay', 10_000, 100_000, step=10_000)
    
    # 将超参数设置到Config类中
    Config.LEARNING_RATE = learning_rate
    Config.BATCH_SIZE = batch_size
    Config.CNN_ALL_GRID_CONFIG['feature_dim'] = cnn_all_dim
    Config.CNN_LOCAL_CONFIG['feature_dim'] = cnn_local_dim
    Config.HIDDEN_LAYERS = [int(base_size * (size_decay ** i)) for i in range(layer_count)]
    Config.GAMMA = gamma
    Config.TARGET_UPDATE = target_update
    Config.FOOD_REWARD = food_reward
    Config.COLLISION_PENALTY = death_penalty
    Config.PROGRESS_REWARD = progress_reward
    Config.STEP_PENALTY = step_penalty
    Config.EPS_DECAY = eps_decay
    
    # 使用多进程并行训练并返回平均分数
    parallel = 5
    num_episodes = 2
    with mp.Pool(processes=parallel) as pool:
        args = [(num_episodes, False, False) for _ in range(parallel)]
        scores = pool.map(train_wrapper, args)
    score = sum(scores) / len(scores)
    
    # 记录超参数优化过程信息
    log_file = "hyperparameter_optimization_log.txt"
    try:
        current_best_score = trial.study.best_value
    except ValueError:
        current_best_score = score

    log_line = f"当前超参数组合为：{trial.params}\n该组合评分为：{score:.4f}，当前历史最高评分：{max(current_best_score, score):.4f}"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_line + '\n')
    print(log_line)
    
    return score


import json
import datetime
from optuna.trial import FrozenTrial, TrialState
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

def load_initial_trials(file_path):
    """加载已知超参数组合及其分数作为初始试验"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            initial_data = json.load(f)
            
        trials = []
        for i, entry in enumerate(initial_data):
            # 创建参数分布字典
            distributions = {}
            for param_name, param_value in entry['params'].items():
                if isinstance(param_value, float):
                    # 对于浮点数参数，我们需要确定其范围
                    # 这里我们使用一个合理的默认范围
                    if param_name == 'learning_rate':
                        distributions[param_name] = FloatDistribution(1e-5, 1e-3, log=True)
                    elif param_name in ['size_decay', 'gamma']:
                        distributions[param_name] = FloatDistribution(0.0, 1.0)
                    elif param_name in ['food_reward', 'death_penalty', 'progress_reward', 'step_penalty']:
                        distributions[param_name] = FloatDistribution(-100.0, 100.0)
                    else:
                        distributions[param_name] = FloatDistribution(0.0, 1.0)
                elif isinstance(param_value, int):
                    # 对于整数参数，我们需要确定其范围
                    if param_name == 'batch_size':
                        distributions[param_name] = CategoricalDistribution([32, 64, 128, 256, 512])
                    elif param_name == 'layer_count':
                        distributions[param_name] = IntDistribution(2, 5)
                    elif param_name == 'base_size':
                        distributions[param_name] = CategoricalDistribution([128, 256, 512, 1024])
                    elif param_name == 'target_update':
                        distributions[param_name] = IntDistribution(50, 500)
                    elif param_name == 'eps_decay':
                        distributions[param_name] = IntDistribution(10_000, 100_000)
                    elif param_name == 'cnn_all_dim':
                        distributions[param_name] = CategoricalDistribution([32, 64, 128, 256])
                    elif param_name == 'cnn_local_dim':
                        distributions[param_name] = CategoricalDistribution([16, 32, 64, 128])
                    else:
                        distributions[param_name] = IntDistribution(0, 1000)
                elif isinstance(param_value, str):
                    # 对于字符串参数，我们假设它们是分类的
                    # 这里我们使用一个占位符，实际应用中需要根据具体情况调整
                    distributions[param_name] = CategoricalDistribution([param_value])
                else:
                    # 对于其他类型，我们使用一个通用的分布
                    distributions[param_name] = FloatDistribution(0.0, 1.0)
            
            # 创建时间戳
            import datetime
            now = datetime.datetime.now()
            
            trial = FrozenTrial(
                number=entry['trial_number'],
                state=TrialState.COMPLETE,
                value=entry['score'],
                datetime_start=now,
                datetime_complete=now,
                params=entry['params'],
                distributions=distributions,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=entry['trial_number']
            )
            trials.append(trial)
        return trials
    except FileNotFoundError:
        print(f"警告: 未找到初始超参数文件 {file_path}，将使用默认搜索策略")
        return []
    except Exception as e:
        print(f"加载初始超参数时出错: {str(e)}")
        return []

if __name__ == '__main__':
    # 创建Optuna研究
    study = optuna.create_study(
        direction='maximize'
    )
    
    # 加载并添加已知超参数组合
    initial_trials = load_initial_trials('initial_hyperparameters.json')
    print(f"成功读取 {len(initial_trials)} 条先验知识")
    for trial in initial_trials:
        study.add_trial(trial)
    
    def objective_with_callback(trial):
        # 调用原始的目标函数
        score = objective(trial)
        
        return score
    
    # 运行优化
    study.optimize(objective_with_callback, n_trials=1000, show_progress_bar=True)
    
    # 打印最佳结果
    print("\n" + "="*50)
    print(f"最佳分数: {study.best_value:.2f}")
    print("最佳超参数组合:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    # 保存最佳参数
    with open('best_hyperparameters.txt', 'w') as f:
        f.write(f"最佳分数: {study.best_value:.2f}\n")
        f.write("最佳超参数组合:\n")
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")