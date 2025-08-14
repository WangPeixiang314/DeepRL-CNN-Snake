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
    size_decay = trial.suggest_float('size_decay', 0.5, 0.9)

    # 5. 训练参数
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    target_update = trial.suggest_int('target_update', 50, 500, step=50)

    # 6. 奖励参数
    food_reward = trial.suggest_float('food_reward', 10.0, 30.0)
    death_penalty = trial.suggest_float('death_penalty', -30.0, -10.0)
    progress_reward = trial.suggest_float('progress_reward', 0.0, 5.0)
    step_penalty = trial.suggest_float('step_penalty', -0.5, 0.5)
    
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
    num_episodes = 300
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
from optuna.trial import FrozenTrial, TrialState

def load_initial_trials(file_path):
    """加载已知超参数组合及其分数作为初始试验"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            initial_data = json.load(f)
            
        trials = []
        for i, entry in enumerate(initial_data):
            trial = FrozenTrial(
                number=i,
                state=TrialState.COMPLETE,
                params=entry['params'],
                value=entry['score'],
                user_attrs={},
                system_attrs={},
                intermediate_values={},
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
    for trial in initial_trials:
        study.add_trial(trial)
    
    # 运行优化
    study.optimize(objective, n_trials=1000, show_progress_bar=True)
    
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
    
    # 可视化优化过程
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()
    
    fig = optuna.visualization.plot_param_importances(study)
    fig.show()