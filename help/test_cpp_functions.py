import sys
import os
import numpy as np

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入C++扩展模块
try:
    import help_cpp
    print("✓ 成功导入help_cpp模块！")
except ImportError as e:
    print(f"✗ 导入help_cpp模块失败: {e}")
    print("请确保你已经成功编译了C++扩展模块")
    sys.exit(1)

# 测试is_collision_py函数
def test_is_collision():
    print("\n测试is_collision_py函数...")
    try:
        # 创建一个简单的蛇（坐标是元组）
        snake = [(10, 10), (9, 10), (8, 10)]
        width, height = 20, 20
        
        # 测试正常情况（不碰撞）
        result1 = help_cpp.is_collision_py(snake, width, height, (11, 10))
        print(f"  测试1（不碰撞）: {'通过' if not result1 else '失败'}")
        
        # 测试边界碰撞
        result2 = help_cpp.is_collision_py(snake, width, height, (20, 10))
        print(f"  测试2（边界碰撞）: {'通过' if result2 else '失败'}")
        
        # 测试自身碰撞
        result3 = help_cpp.is_collision_py(snake, width, height, (9, 10))
        print(f"  测试3（自身碰撞）: {'通过' if result3 else '失败'}")
    except Exception as e:
        print(f"  测试失败: {e}")

# 测试distance_py函数
def test_distance():
    print("\n测试distance_py函数...")
    try:
        # 测试曼哈顿距离
        point1 = (0, 0)
        point2 = (3, 4)
        result = help_cpp.distance_py(point1, point2)
        expected = 7
        print(f"  曼哈顿距离测试: {'通过' if result == expected else '失败'} (结果: {result}, 预期: {expected})")
    except Exception as e:
        print(f"  测试失败: {e}")

# 测试update_direction_py函数
def test_update_direction():
    print("\n测试update_direction_py函数...")
    try:
        # 方向: RIGHT=(1,0), DOWN=(0,1), LEFT=(-1,0), UP=(0,-1)
        current_dir = (1, 0)  # 向右
        
        # 测试直行
        result1 = help_cpp.update_direction_py(current_dir, 0)
        print(f"  测试1（直行）: {'通过' if result1 == (1, 0) else '失败'} {result1}")
        
        # 测试右转
        result2 = help_cpp.update_direction_py(current_dir, 1)
        print(f"  测试2（右转）: {'通过' if result2 == (0, 1) else '失败'} {result2}")
        
        # 测试左转
        result3 = help_cpp.update_direction_py(current_dir, 2)
        print(f"  测试3（左转）: {'通过' if result3 == (0, -1) else '失败'} {result3}")
    except Exception as e:
        print(f"  测试失败: {e}")

# 测试get_local_grid_view_py函数
def test_get_local_grid_view():
    print("\n测试get_local_grid_view_py函数...")
    try:
        snake = [(10, 10), (9, 10), (8, 10)]
        head = (10, 10)
        food = (12, 10)
        width, height = 20, 20
        BLOCK_SIZE = 1
        
        result = help_cpp.get_local_grid_view_py(head, snake, food, width, height, BLOCK_SIZE)
        # 检查返回的是否是numpy数组
        if hasattr(result, 'shape'):
            print(f"  返回类型测试: 通过 (numpy数组)")
            print(f"  数组形状: {result.shape}")
            # 检查是否是1维36元素的数组（因为C++中是1维数组）
            success = len(result.shape) == 1 and result.shape[0] == 36
            print(f"  数组大小测试: {'通过' if success else '失败'} (长度: {result.shape[0] if hasattr(result, 'shape') else 0})")
        else:
            print(f"  返回类型测试: 失败 (不是numpy数组)")
    except Exception as e:
        print(f"  测试失败: {e}")

# 测试safe_action_py函数
def test_safe_action():
    print("\n测试safe_action_py函数...")
    try:
        # 正确的参数格式: q_values, danger_signals, collision_penalty, state
        import numpy as np
        q_values = np.array([0.1, 0.2, 0.3], dtype=np.float32)  # 直行、右转、左转的Q值
        danger_signals = np.array([0.0, 0.8, 0.0], dtype=np.float32)  # 中间动作有危险
        collision_penalty = -10.0
        state = np.random.rand(10).astype(np.float32)  # 随机状态向量
        
        result = help_cpp.safe_action_py(q_values, danger_signals, collision_penalty, state)
        print(f"  函数调用测试: 通过")
        print(f"  返回类型测试: {'通过' if isinstance(result, tuple) and len(result) == 2 else '失败'} (返回类型: {type(result)})")
        if isinstance(result, tuple):
            print(f"  选择的动作: {result[0]}")
            print(f"  危险动作列表: {result[1]}")
            # 危险动作应该包含索引1
            print(f"  危险动作检测: {'通过' if 1 in result[1] else '失败'}")
    except Exception as e:
        print(f"  函数调用测试: 失败 (错误: {str(e)})" )
    

# 测试step_logic_py函数
def test_step_logic():
    print("\n测试step_logic_py函数...")
    try:
        snake = [(10, 10), (9, 10), (8, 10)]
        head = (10, 10)
        direction = (1, 0)  # 向右
        food = (12, 10)
        steps_since_food = 0
        score = 0
        prev_distance = 2
        width, height = 20, 20
        BLOCK_SIZE = 1
        action = 0  # 直行
        MAX_STEPS_WITHOUT_FOOD = 100
        FOOD_REWARD = 10.0
        COLLISION_PENALTY = -50.0
        PROGRESS_REWARD = 0.1
        STEP_PENALTY = -0.1
        
        result = help_cpp.step_logic_py(
            snake, head, direction, food, steps_since_food, score, prev_distance,
            width, height, BLOCK_SIZE, action, MAX_STEPS_WITHOUT_FOOD,
            FOOD_REWARD, COLLISION_PENALTY, PROGRESS_REWARD, STEP_PENALTY
        )
        
        # 检查返回值数量（应该是9个返回值）
        success = len(result) == 9
        print(f"  返回值数量测试: {'通过' if success else '失败'} (返回值数量: {len(result)})")
        print(f"  新蛇长度: {len(result[0])}")
        print(f"  新分数: {result[5]}")
        print(f"  奖励值: {result[6]}")
        print(f"  游戏结束标志: {result[7]}")
    except Exception as e:
        print(f"  测试失败: {e}")

# 运行所有测试
if __name__ == "__main__":
    print("===== C++扩展模块函数测试 =====")
    
    # 调用所有测试函数
    test_is_collision()
    test_distance()
    test_update_direction()
    test_get_local_grid_view()
    test_safe_action()
    test_step_logic()
    
    print("\n===== 测试完成 =====")