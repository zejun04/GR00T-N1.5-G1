# GR00T 观测数据可视化工具

这个工具可以实时可视化从Isaac Lab仿真环境发送给GR00T推理服务的观测图像数据。

## 功能特点

- **实时可视化**: 显示从仿真环境中捕获的实时相机图像
- **多相机支持**: 支持同时显示多个相机（左臂高位、左臂腕部、右臂腕部）
- **任务信息显示**: 显示当前执行的任务描述
- **时间戳**: 显示数据更新的时间戳
- **非阻塞**: 可视化在后台线程运行，不影响仿真性能

## 使用方法

### 1. 启动GR00T推理服务

首先确保GR00T推理服务正在运行：

```bash
# 在Isaac-GR00T目录下
cd Isaac-GR00T
python scripts/inference_service.py --server --http-server --port 8000 --embodiment_tag gr1 --data_config fourier_gr1_arms_waist
```

### 2. 启动仿真环境（带可视化）

```bash
# 在g1_xr_teleoperate/unitree_sim_isaaclab目录下
cd g1_xr_teleoperate/unitree_sim_isaaclab

# 默认启用可视化
python sim_main.py --action_source gr00t --task Isaac-PickPlace-G129-Dex3-Joint --enable_dex3_dds --robot_type g129

# 或者明确指定启用可视化
python sim_main.py --action_source gr00t --task Isaac-PickPlace-G129-Dex3-Joint --enable_dex3_dds --robot_type g129 --enable_visualization

# 如果想禁用可视化
python sim_main.py --action_source gr00t --task Isaac-PickPlace-G129-Dex3-Joint --enable_dex3_dds --robot_type g129 --no-enable_visualization
```

### 3. 可视化窗口操作

启动仿真后，会自动打开一个名为"GR00T 观测数据可视化"的窗口：

- **窗口布局**: 以网格形式显示所有活跃的相机图像
- **相机标签**: 每个相机图像上方显示相机名称
- **状态信息**: 底部显示时间戳和当前任务描述
- **退出**: 按 `q` 键关闭可视化窗口（仿真继续运行）

## 可视化窗口内容

### 显示的相机
- **左臂高位相机 (cam_left_high)**: 通常是前置相机，提供全局视野
- **左臂腕部相机 (cam_left_wrist)**: 安装在左臂末端的相机
- **右臂腕部相机 (cam_right_wrist)**: 安装在右臂末端的相机

### 图像处理
- 自动调整图像尺寸为统一显示格式
- 自动处理RGB到BGR转换（OpenCV格式）
- 支持不同数据类型的图像（uint8, float等）
- 处理带Alpha通道的RGBA图像

## 测试可视化功能

运行独立的测试脚本来验证可视化功能：

```bash
python test_visualization.py
```

这个脚本会：
- 创建测试观测数据
- 启动可视化窗口
- 模拟数据更新过程
- 验证可视化功能是否正常工作

## 技术实现

### 架构设计
1. **GR00TActionProvider** 集成可视化功能
2. **后台线程**: 可视化在独立线程运行，不阻塞仿真主循环
3. **线程安全**: 使用锁机制保护观测数据的并发访问
4. **内存管理**: 自动清理OpenCV窗口资源

### 数据流
```
仿真环境 -> 相机传感器 -> GR00TActionProvider.prepare_observation()
    ↓
观测数据准备 -> 保存到可视化缓冲区 -> 后台可视化线程
    ↓
HTTP请求 -> GR00T推理服务 -> 返回动作 -> 应用到仿真环境
```

### 性能考虑
- 可视化更新频率：100ms（10 FPS）
- 图像处理：只在需要时进行格式转换
- 内存使用：重用显示缓冲区，避免频繁分配

## 故障排除

### 常见问题

1. **可视化窗口不显示**
   - 检查OpenCV是否正确安装：`pip install opencv-python`
   - 确认系统支持GUI显示
   - 检查是否有DISPLAY环境变量（Linux）

2. **图像显示异常**
   - 检查相机配置是否正确启用
   - 确认图像数据格式正确
   - 查看控制台错误信息

3. **性能问题**
   - 如可视化影响仿真性能，可以通过`--no-enable_visualization`禁用
   - 可视化线程是守护线程，程序退出时会自动清理

4. **相机数据为空**
   - 确认任务配置中包含所需的相机
   - 检查相机是否在`--camera_include`列表中
   - 确认相机传感器正确初始化

### 调试信息

仿真启动时会显示以下相关信息：
```
📺 启动观测可视化窗口
✅ GR00T Action Provider initialized successfully
🟡 环境动作空间: ...
```

可视化过程中会显示：
```
🔍 机器人状态已更新: left_arm: shape=(1, 7), range=[-0.500, 0.800]
```

## 扩展功能

### 添加新相机
在 `GR00TActionProvider._get_camera_observations()` 中添加新的相机配置：

```python
camera_mapping = {
    'new_camera': 'cam_new',
    # ...
}
```

### 自定义显示布局
修改 `_create_visualization_image()` 方法来自定义网格布局和显示样式。

### 添加状态信息
在可视化图像上添加更多状态信息，如关节角度、推理延迟等。

## 版本历史

- **v1.0**: 初始版本，支持基本的多相机可视化
- **v1.1**: 添加线程安全机制和错误处理
- **v1.2**: 优化性能和内存使用

---

如有问题或建议，请查看控制台输出信息或检查GR00TActionProvider的日志。


