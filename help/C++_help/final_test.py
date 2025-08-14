import ctypes
import os
import sys
import numpy as np

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(current_dir, "help_cpp.dll")

print(f"🐍 Testing Snake Game C++ DLL")
print(f"Loading from: {dll_path}")

if not os.path.exists(dll_path):
    print(f"❌ Error: DLL not found at {dll_path}")
    sys.exit(1)

# 加载DLL
lib = ctypes.CDLL(dll_path)

print("✅ DLL loaded successfully!")

# 测试函数定义
def test_distance():
    """测试距离计算"""
    lib.distance_py.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.distance_py.restype = ctypes.c_int
    
    result = lib.distance_py(0, 0, 3, 4)
    expected = 7
    assert result == expected, f"Expected {expected}, got {result}"
    print("✅ distance_py test passed")

def test_collision():
    """测试碰撞检测"""
    lib.is_collision_py.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.is_collision_py.restype = ctypes.c_bool
    
    # 创建测试数据
    snake_x = (ctypes.c_int * 3)(100, 80, 60)
    snake_y = (ctypes.c_int * 3)(100, 100, 100)
    
    # 测试边界碰撞
    result = lib.is_collision_py(snake_x, snake_y, 3, 100, 100, 150, 50)
    assert result == True, "Should detect boundary collision"
    
    # 测试自身碰撞
    result = lib.is_collision_py(snake_x, snake_y, 3, 200, 200, 80, 100)
    assert result == True, "Should detect self collision"
    
    # 测试无碰撞
    result = lib.is_collision_py(snake_x, snake_y, 3, 200, 200, 120, 100)
    assert result == False, "Should not detect collision"
    
    print("✅ is_collision_py test passed")

def test_random_position():
    """测试随机位置生成"""
    lib.random_position_py.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                     ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.random_position_py.restype = None
    
    out_x = ctypes.c_int()
    out_y = ctypes.c_int()
    
    lib.random_position_py(800, 600, 20, ctypes.byref(out_x), ctypes.byref(out_y))
    
    assert 0 <= out_x.value <= 800, f"Invalid x: {out_x.value}"
    assert 0 <= out_y.value <= 600, f"Invalid y: {out_y.value}"
    print("✅ random_position_py test passed")

def test_place_food():
    """测试食物放置"""
    lib.place_food_py.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
    ]
    lib.place_food_py.restype = None
    
    snake_x = (ctypes.c_int * 3)(100, 80, 60)
    snake_y = (ctypes.c_int * 3)(100, 100, 100)
    
    food_x = ctypes.c_int()
    food_y = ctypes.c_int()
    
    lib.place_food_py(snake_x, snake_y, 3, 800, 600, 20, 
                     ctypes.byref(food_x), ctypes.byref(food_y))
    
    # 检查食物不在蛇身上
    for i in range(3):
        assert not (food_x.value == snake_x[i] and food_y.value == snake_y[i])
    
    print("✅ place_food_py test passed")

def test_direction_update():
    """测试方向更新"""
    lib.update_direction_py.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.update_direction_py.restype = None
    
    new_dx = ctypes.c_int()
    new_dy = ctypes.c_int()
    
    # 测试直行
    lib.update_direction_py(1, 0, 0, ctypes.byref(new_dx), ctypes.byref(new_dy))
    assert new_dx.value == 1 and new_dy.value == 0
    
    # 测试右转
    lib.update_direction_py(1, 0, 1, ctypes.byref(new_dx), ctypes.byref(new_dy))
    assert new_dx.value == 0 and new_dy.value == 1
    
    # 测试左转
    lib.update_direction_py(1, 0, 2, ctypes.byref(new_dx), ctypes.byref(new_dy))
    assert new_dx.value == 0 and new_dy.value == -1
    
    print("✅ update_direction_py test passed")

def test_manhattan_distance():
    """测试曼哈顿距离"""
    lib.get_manhattan_distance_py.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.get_manhattan_distance_py.restype = ctypes.c_float
    
    result = lib.get_manhattan_distance_py(100, 100, 200, 150, 800, 600)
    expected = (100 + 50) / (800 + 600)  # 150 / 1400
    assert abs(result - expected) < 1e-6
    print("✅ get_manhattan_distance_py test passed")

def test_safe_action():
    """测试安全动作选择"""
    lib.safe_action_py.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    lib.safe_action_py.restype = ctypes.c_int
    
    q_values = (ctypes.c_float * 3)(1.0, 0.5, 0.8)
    danger_signals = (ctypes.c_float * 3)(0.0, 1.0, 0.0)
    state = (ctypes.c_float * 10)()
    
    result = lib.safe_action_py(
        q_values, 3, danger_signals, 3, -10.0, state, 10
    )
    
    # 应该避免危险动作(1)，选择安全的最高Q值动作(0)
    assert result == 0, f"Expected 0, got {result}"
    print("✅ safe_action_py test passed")

def test_boundary_distances():
    """测试边界距离计算"""
    lib.get_boundary_distances_py.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.get_boundary_distances_py.restype = None
    
    output = (ctypes.c_float * 4)()
    lib.get_boundary_distances_py(400, 300, 800, 600, output)
    
    expected = [400/800, 400/800, 300/600, 300/600]  # [0.5, 0.5, 0.5, 0.5]
    for i in range(4):
        assert abs(output[i] - expected[i]) < 1e-6
    print("✅ get_boundary_distances_py test passed")

def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*50)
    print("🧪 Starting comprehensive tests...")
    print("="*50)
    
    try:
        test_distance()
        test_collision()
        test_random_position()
        test_place_food()
        test_direction_update()
        test_manhattan_distance()
        test_safe_action()
        test_boundary_distances()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("✅ C++ DLL is working perfectly!")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()