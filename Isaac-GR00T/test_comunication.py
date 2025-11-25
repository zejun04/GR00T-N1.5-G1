# test_gr00t_direct_numpy.py

import requests
import numpy as np
import json
import sys

# 添加GR00T项目路径
sys.path.append('/home/shenlan/GR00T-VLA/Isaac-GR00T')

def test_with_json_numpy():
    """使用json-numpy库来正确处理numpy数组"""
    
    try:
        # 导入json_numpy来正确处理numpy数组
        import json_numpy
        json_numpy.patch()
        print("✅ json_numpy已加载")
    except ImportError:
        print("❌ 请先安装json_numpy: pip install json-numpy")
        return False
    
    # 创建测试数据
    test_obs = {
        "video.rs_view": np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8),
        "state.left_arm": np.random.rand(1, 7).astype(np.float32),
        "state.right_arm": np.random.rand(1, 7).astype(np.float32),
        "state.left_hand": np.random.rand(1, 7).astype(np.float32),
        "state.right_hand": np.random.rand(1, 7).astype(np.float32),
        "state.waist": np.random.rand(1, 3).astype(np.float32),
        "annotation.human.action.task_description": ["Test task"]
    }
    
    print("📊 测试数据 (numpy格式):")
    for key, value in test_obs.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {value}")
    
    try:
        # 使用json_numpy序列化
        import json_numpy
        json_data = json_numpy.dumps({"observation": test_obs})
        
        # 发送请求
        headers = {'Content-Type': 'application/json'}
        response = requests.post(
            "http://localhost:8000/act",
            data=json_data,
            headers=headers,
            timeout=10.0
        )
        
        if response.status_code == 200:
            # 使用json_numpy反序列化响应
            action_data = json_numpy.loads(response.text)
            print("✅ 通信成功!")
            for key, value in action_data.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            return True
        else:
            print(f"❌ 失败: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        return False

if __name__ == "__main__":
    print("🧪 使用json_numpy测试GR00T通信...")
    test_with_json_numpy()
