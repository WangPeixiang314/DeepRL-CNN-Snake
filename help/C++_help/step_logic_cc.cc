#include "help_cpp.h"

std::tuple<
    std::vector<std::tuple<int, int>>,
    std::tuple<int, int>,
    std::tuple<int, int>,
    std::tuple<int, int>,
    int,
    int,
    float,
    bool,
    int
> step_logic_py(
    const std::vector<std::tuple<int, int>>& snake,
    const std::tuple<int, int>& head,
    const std::tuple<int, int>& direction_vec,
    const std::tuple<int, int>& food,
    int steps_since_food,
    int score,
    int prev_distance,
    int width,
    int height,
    int BLOCK_SIZE,
    int action,
    int MAX_STEPS_WITHOUT_FOOD,
    float FOOD_REWARD,
    float COLLISION_PENALTY,
    float PROGRESS_REWARD,
    float STEP_PENALTY
) {
    // 1. 更新方向
    std::tuple<int, int> new_direction_vec = update_direction_py(direction_vec, action);

    // 2. 移动蛇头
    int dx = std::get<0>(new_direction_vec);
    int dy = std::get<1>(new_direction_vec);
    int new_head_x = std::get<0>(head) + dx * BLOCK_SIZE;
    int new_head_y = std::get<1>(head) + dy * BLOCK_SIZE;
    std::tuple<int, int> new_head = std::make_tuple(new_head_x, new_head_y);

    std::vector<std::tuple<int, int>> new_snake = snake;
    new_snake.insert(new_snake.begin(), new_head);

    // 3. 检查游戏结束
    bool done = false;
    float reward = 0.0f;
    int new_steps_since_food = steps_since_food;
    int new_score = score;
    std::tuple<int, int> new_food = food;
    int new_prev_distance = prev_distance;

    if (is_collision_py(new_snake, width, height, new_head) || steps_since_food >= MAX_STEPS_WITHOUT_FOOD) {
        reward = COLLISION_PENALTY;
        done = true;
    } else if (new_head == food) {
        // 4. 检查吃到食物
        // 根据蛇的长度动态计算得分
        int snake_length = new_snake.size();
        // 假设基础得分为1，每增加一个单位长度，得分增加0.5
        float dynamic_score = 1.0f + (snake_length - 3) * 0.5f;  // 初始长度为3
        new_score = score + dynamic_score;
        
        new_food = place_food_py(new_snake, width, height, BLOCK_SIZE);
        // 检查是否游戏胜利（无法放置新食物）
        if (std::get<0>(new_food) == -1 && std::get<1>(new_food) == -1) {
            new_steps_since_food = 0;
            // 满分奖励：基础食物奖励 + 动态得分 + 额外奖励
            reward = FOOD_REWARD + dynamic_score + 1000.0f;  // 游戏胜利额外奖励
            new_prev_distance = 0;
            done = true;  // 游戏胜利
        } else {
            new_steps_since_food = 0;
            // 奖励也使用动态得分
            reward = FOOD_REWARD + dynamic_score;
            new_prev_distance = distance_py(new_head, new_food);
        }
    } else {
        // 5. 没吃到食物
        new_snake.pop_back();
        new_steps_since_food = steps_since_food + 1;

        // 计算距离奖励
        int distance = distance_py(new_head, food);
        reward = PROGRESS_REWARD * (prev_distance - distance);
        new_prev_distance = distance;

        // 添加步数惩罚
        reward += STEP_PENALTY;
    }

    return std::make_tuple(
        new_snake,
        new_head,
        new_direction_vec,
        new_food,
        new_steps_since_food,
        new_score,
        reward,
        done,
        new_prev_distance
    );
}