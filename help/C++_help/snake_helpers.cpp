// Snake Game Helper Functions - C Interface
// Optimized for Windows DLL with static linking

#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>

// 静态随机数生成器
static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());

extern "C" {

// 基本距离计算
__declspec(dllexport) int distance_py(int x1, int y1, int x2, int y2) {
    return std::abs(x1 - x2) + std::abs(y1 - y2);
}

// 碰撞检测
__declspec(dllexport) bool is_collision_py(
    const int* snake_x, const int* snake_y, int snake_size,
    int width, int height, int check_x, int check_y
) {
    // 边界检查
    if (check_x >= width || check_x < 0 || check_y >= height || check_y < 0) {
        return true;
    }
    
    // 自身碰撞检查
    for (int i = 1; i < snake_size; ++i) {
        if (snake_x[i] == check_x && snake_y[i] == check_y) {
            return true;
        }
    }
    
    return false;
}

// 随机位置生成
__declspec(dllexport) void random_position_py(int width, int height, int block_size, int* out_x, int* out_y) {
    int max_x = ((width - block_size) / block_size) + 1;
    int max_y = ((height - block_size) / block_size) + 1;
    
    std::uniform_int_distribution<int> dist_x(0, std::max(0, max_x - 1));
    std::uniform_int_distribution<int> dist_y(0, std::max(0, max_y - 1));
    
    *out_x = dist_x(rng) * block_size;
    *out_y = dist_y(rng) * block_size;
}

// 放置食物
__declspec(dllexport) void place_food_py(
    const int* snake_x, const int* snake_y, int snake_size,
    int width, int height, int block_size, int* out_x, int* out_y
) {
    int food_x, food_y;
    bool valid_position = false;
    
    while (!valid_position) {
        random_position_py(width, height, block_size, &food_x, &food_y);
        valid_position = true;
        
        for (int i = 0; i < snake_size; ++i) {
            if (snake_x[i] == food_x && snake_y[i] == food_y) {
                valid_position = false;
                break;
            }
        }
    }
    
    *out_x = food_x;
    *out_y = food_y;
}

// 方向更新
__declspec(dllexport) void update_direction_py(int current_dx, int current_dy, int action, int* new_dx, int* new_dy) {
    static const int directions[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
    
    int current_idx = -1;
    for (int i = 0; i < 4; ++i) {
        if (directions[i][0] == current_dx && directions[i][1] == current_dy) {
            current_idx = i;
            break;
        }
    }
    
    if (current_idx == -1) {
        *new_dx = current_dx;
        *new_dy = current_dy;
        return;
    }
    
    int new_idx;
    if (action == 0) {  // 直行
        new_idx = current_idx;
    } else if (action == 1) {  // 右转
        new_idx = (current_idx + 1) % 4;
    } else if (action == 2) {  // 左转
        new_idx = (current_idx - 1 + 4) % 4;
    } else {
        new_idx = current_idx;
    }
    
    *new_dx = directions[new_idx][0];
    *new_dy = directions[new_idx][1];
}

// 游戏逻辑核心
__declspec(dllexport) void step_logic_py(
    const int* snake_x, const int* snake_y, int snake_size,
    int head_x, int head_y,
    int direction_dx, int direction_dy,
    int food_x, int food_y,
    int steps_since_food, int score, float prev_distance,
    int width, int height, int block_size, int action,
    int max_steps_without_food, float food_reward, float collision_penalty,
    float progress_reward, float step_penalty,
    // 输出参数
    int* out_snake_x, int* out_snake_y, int* out_snake_size,
    int* out_head_x, int* out_head_y,
    int* out_direction_dx, int* out_direction_dy,
    int* out_food_x, int* out_food_y,
    int* out_steps_since_food, int* out_score, float* out_reward,
    bool* out_done, float* out_prev_distance
) {
    // 更新方向
    int new_dx, new_dy;
    update_direction_py(direction_dx, direction_dy, action, &new_dx, &new_dy);
    
    // 移动蛇头
    int new_head_x = head_x + new_dx * block_size;
    int new_head_y = head_y + new_dy * block_size;
    
    // 复制蛇身
    for (int i = 0; i < snake_size; ++i) {
        out_snake_x[i] = snake_x[i];
        out_snake_y[i] = snake_y[i];
    }
    
    // 添加新头部
    out_snake_x[snake_size] = new_head_x;
    out_snake_y[snake_size] = new_head_y;
    *out_snake_size = snake_size + 1;
    
    // 检查游戏结束
    if (is_collision_py(out_snake_x, out_snake_y, *out_snake_size, width, height, new_head_x, new_head_y) || 
        steps_since_food >= max_steps_without_food) {
        *out_reward = collision_penalty;
        *out_done = true;
        *out_food_x = food_x;
        *out_food_y = food_y;
        *out_steps_since_food = steps_since_food;
        *out_score = score;
        *out_prev_distance = prev_distance;
        return;
    }
    
    // 检查吃到食物
    if (new_head_x == food_x && new_head_y == food_y) {
        *out_score = score + 1;
        place_food_py(out_snake_x, out_snake_y, *out_snake_size, width, height, block_size, out_food_x, out_food_y);
        *out_steps_since_food = 0;
        *out_reward = food_reward;
        *out_prev_distance = static_cast<float>(distance_py(new_head_x, new_head_y, *out_food_x, *out_food_y));
        *out_done = false;
    } else {
        // 移除尾部
        *out_snake_size = snake_size;
        *out_steps_since_food = steps_since_food + 1;
        *out_score = score;
        *out_food_x = food_x;
        *out_food_y = food_y;
        
        // 计算距离奖励
        int distance = distance_py(new_head_x, new_head_y, food_x, food_y);
        *out_reward = progress_reward * (prev_distance - distance);
        *out_prev_distance = static_cast<float>(distance);
        
        // 添加步数惩罚
        *out_reward += step_penalty;
        *out_done = false;
    }
    
    *out_head_x = new_head_x;
    *out_head_y = new_head_y;
    *out_direction_dx = new_dx;
    *out_direction_dy = new_dy;
}

// SumTree操作
__declspec(dllexport) void propagate_py(float* tree, int tree_size, int idx, float change) {
    int parent = (idx - 1) / 2;
    while (parent >= 0) {
        tree[parent] += change;
        if (parent == 0) break;
        parent = (parent - 1) / 2;
    }
}

__declspec(dllexport) int retrieve_py(const float* tree, int capacity, float s) {
    int idx = 0;
    while (idx < capacity - 1) {
        int left = 2 * idx + 1;
        if (left >= capacity) break;
        
        if (s <= tree[left]) {
            idx = left;
        } else {
            s -= tree[left];
            idx = left + 1;
        }
    }
    return idx;
}

__declspec(dllexport) void batch_retrieve_py(const float* tree, int capacity, const float* s_values, int s_size, int* output) {
    for (int i = 0; i < s_size; ++i) {
        float s = s_values[i];
        int idx = 0;
        while (idx < capacity - 1) {
            int left = 2 * idx + 1;
            if (left >= capacity) break;
            
            if (s <= tree[left]) {
                idx = left;
            } else {
                s -= tree[left];
                idx = left + 1;
            }
        }
        output[i] = idx;
    }
}

// 安全动作选择
__declspec(dllexport) int safe_action_py(
    const float* q_values, int q_size,
    const float* danger_signals, int danger_size,
    float collision_penalty,
    const float* state, int state_size
) {
    if (q_size != 3 || danger_size != 3) {
        return 0;
    }
    
    // 创建安全动作掩码
    bool safe_mask[3] = {true, true, true};
    
    // 标记危险动作
    for (int i = 0; i < 3; ++i) {
        if (danger_signals[i] > 0.5f) {
            safe_mask[i] = false;
        }
    }
    
    // 选择安全动作中Q值最高的
    int best_action = 0;
    float best_q_value = -1e10f;
    
    for (int i = 0; i < 3; ++i) {
        if (safe_mask[i] && q_values[i] > best_q_value) {
            best_q_value = q_values[i];
            best_action = i;
        }
    }
    
    // 如果没有安全动作，选择原始Q值最高的
    if (best_q_value == -1e10f) {
        for (int i = 0; i < 3; ++i) {
            if (q_values[i] > best_q_value) {
                best_q_value = q_values[i];
                best_action = i;
            }
        }
    }
    
    return best_action;
}

// 特征提取函数
__declspec(dllexport) void get_head_pos_py(int head_x, int head_y, int width, int height, float* output) {
    output[0] = static_cast<float>(head_x) / width;
    output[1] = static_cast<float>(head_y) / height;
}

__declspec(dllexport) void get_food_pos_py(int food_x, int food_y, int width, int height, float* output) {
    output[0] = static_cast<float>(food_x) / width;
    output[1] = static_cast<float>(food_y) / height;
}

__declspec(dllexport) void get_relative_distance_py(int head_x, int head_y, int food_x, int food_y, int width, int height, float* output) {
    output[0] = static_cast<float>(food_x - head_x) / width;
    output[1] = static_cast<float>(food_y - head_y) / height;
}

__declspec(dllexport) float get_manhattan_distance_py(int head_x, int head_y, int food_x, int food_y, int width, int height) {
    int dist = std::abs(food_x - head_x) + std::abs(food_y - head_y);
    return static_cast<float>(dist) / (width + height);
}

__declspec(dllexport) void get_direction_onehot_py(int direction, float* output) {
    for (int i = 0; i < 4; ++i) {
        output[i] = (i == direction) ? 1.0f : 0.0f;
    }
}

__declspec(dllexport) void get_boundary_distances_py(int head_x, int head_y, int width, int height, float* output) {
    output[0] = static_cast<float>(head_x) / width;                    // 左边界
    output[1] = static_cast<float>(width - head_x) / width;            // 右边界
    output[2] = static_cast<float>(head_y) / height;                   // 上边界
    output[3] = static_cast<float>(height - head_y) / height;           // 下边界
}

__declspec(dllexport) float get_snake_length_py(int snake_size, float grid_area) {
    return static_cast<float>(snake_size) / grid_area;
}

__declspec(dllexport) float get_free_space_ratio_py(int snake_size, float grid_area) {
    return (grid_area - snake_size) / grid_area;
}

__declspec(dllexport) void get_local_grid_view_py(
    int head_x, int head_y,
    const int* snake_x, const int* snake_y, int snake_size,
    int food_x, int food_y,
    int width, int height, int block_size, float* output
) {
    int grid_width = width / block_size;
    int grid_height = height / block_size;
    int center_x = head_x / block_size;
    int center_y = head_y / block_size;
    
    for (int dx = -3; dx < 3; ++dx) {
        for (int dy = -3; dy < 3; ++dy) {
            int grid_x = center_x + dx;
            int grid_y = center_y + dy;
            int idx = (dx + 3) * 6 + (dy + 3);
            
            if (0 <= grid_x && grid_x < grid_width && 
                0 <= grid_y && grid_y < grid_height) {
                int pixel_x = grid_x * block_size;
                int pixel_y = grid_y * block_size;
                
                // 检查是否有蛇身
                bool is_snake = false;
                for (int i = 0; i < snake_size; ++i) {
                    if (snake_x[i] == pixel_x && snake_y[i] == pixel_y) {
                        output[idx] = 1.0f;
                        is_snake = true;
                        break;
                    }
                }
                
                if (!is_snake && pixel_x == food_x && pixel_y == food_y) {
                    output[idx] = 0.5f;
                } else if (!is_snake) {
                    output[idx] = 0.0f;
                }
            } else {
                output[idx] = 1.0f;  // 边界外视为障碍
            }
        }
    }
}

__declspec(dllexport) void get_all_grid_view_py(
    const int* snake_x, const int* snake_y, int snake_size,
    int food_x, int food_y,
    int width, int height, int block_size, float* output
) {
    int grid_width = width / block_size;
    int grid_height = height / block_size;
    int grid_area = grid_width * grid_height;
    
    // 初始化
    for (int i = 0; i < grid_area; ++i) {
        output[i] = 0.0f;
    }
    
    // 填充蛇身
    for (int i = 0; i < snake_size; ++i) {
        int segment_x = snake_x[i] / block_size;
        int segment_y = snake_y[i] / block_size;
        if (0 <= segment_x && segment_x < grid_width && 
            0 <= segment_y && segment_y < grid_height) {
            int idx = segment_y * grid_width + segment_x;
            output[idx] = 1.0f;
        }
    }
    
    // 填充食物
    int food_grid_x = food_x / block_size;
    int food_grid_y = food_y / block_size;
    if (0 <= food_grid_x && food_grid_x < grid_width && 
        0 <= food_grid_y && food_grid_y < grid_height) {
        int idx = food_grid_y * grid_width + food_grid_x;
        output[idx] = 0.5f;
    }
}

__declspec(dllexport) void get_action_history_onehot_py(const int* action_history, int history_size, float* output) {
    for (int i = 0; i < 15; ++i) {
        output[i] = 0.0f;
    }
    
    int start_idx = std::max(0, 5 - history_size);
    
    for (int i = 0; i < std::min(history_size, 5); ++i) {
        int action_idx = (start_idx + i) * 3;
        int action = action_history[history_size - 5 + i];
        if (action >= 0 && action < 3) {
            output[action_idx + action] = 1.0f;
        }
    }
}

} // extern "C"