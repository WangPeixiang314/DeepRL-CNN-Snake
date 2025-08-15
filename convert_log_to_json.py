import json
import re

def parse_log_file(log_file_path, json_file_path):
    """解析日志文件并将其转换为JSON格式"""
    results = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 处理日志文件中的每两行
    for i in range(0, len(lines), 2):
        if i + 1 < len(lines):
            # 提取超参数组合
            params_line = lines[i].strip()
            # 使用正则表达式提取参数字典
            params_match = re.search(r"当前超参数组合为：({.*})", params_line)
            if params_match:
                params_str = params_match.group(1)
                # 将单引号替换为双引号以便JSON解析
                params_str = params_str.replace("'", "\"")
                try:
                    params = json.loads(params_str)
                except json.JSONDecodeError:
                    # 如果JSON解析失败，则手动解析
                    params = {}
                    # 分割字符串并解析键值对
                    pairs = params_str.strip('{}').split(', ')
                    for pair in pairs:
                        if ':' in pair:
                            key, value = pair.split(':', 1)
                            key = key.strip().strip('"')
                            value = value.strip().strip('"')
                            # 尝试转换为适当的类型
                            try:
                                if '.' in value:
                                    params[key] = float(value)
                                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                                    params[key] = int(value)
                                else:
                                    params[key] = value
                            except:
                                params[key] = value
            
            # 提取评分
            score_line = lines[i+1].strip()
            score_match = re.search(r"该组合评分为：([0-9.]+).*当前历史最高评分：([0-9.]+)", score_line)
            if score_match:
                score = float(score_match.group(1))
                best_score = float(score_match.group(2))
                
                # 添加到结果列表
                results.append({
                    "trial_number": len(results),
                    "params": params,
                    "score": score,
                    "best_score_so_far": best_score
                })
    
    # 保存为JSON文件
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"已将日志文件转换为JSON格式，保存到 {json_file_path}")

if __name__ == "__main__":
    parse_log_file("hyperparameter_optimization_log.txt", "hyperparameter_optimization_results.json")