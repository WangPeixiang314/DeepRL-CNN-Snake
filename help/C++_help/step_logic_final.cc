#include "help_cpp.h"
#include <vector>
#include <tuple>

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
    std::tuple<int, int> new_direction_vec = update_direction_py(direction_vec, action);

    int dx = std::get<0>(new_direction_vec);
    int dy = std::get<1>(new_direction_vec);
    int new_head_x = std::get<0>(head) + dx * BLOCK_SIZE;
    int new_head_y = std::get<1>(head) + dy * BLOCK_SIZE;
    std::tuple<int, int> new_head = std::make_tuple(new_head_x, new_head_y);

    std::vector<std::tuple<int, int>> new_snake = snake;
    new_snake.insert(new_snake.begin(), new_head);

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
        new_score = score + 1;
        new_food = place_food_py(new_snake, width, height, BLOCK_SIZE);
        new_steps_since_food = 0;
        reward = FOOD_REWARD;
        new_prev_distance = distance_py(new_head, new_food);
    } else {
        new_snake.pop_back();
        new_steps_since_food = steps_since_food + 1;

        int distance = distance_py(new_head, food);
        reward = PROGRESS_REWARD * (prev_distance - distance);
        new_prev_distance = distance;

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