#!/usr/bin/env python3
"""
GR00T 可视化功能测试脚本

测试观测数据可视化功能是否正常工作
"""

import numpy as np
import cv2
import time
import threading

class TestVisualization:
    def __init__(self):
        self.visualization_running = False
        self.visualization_thread = None
        self.current_observation = None
        self.observation_lock = threading.Lock()

        # 启动可视化
        self._start_visualization()

    def _start_visualization(self):
        """启动可视化线程"""
        self.visualization_running = True
        self.visualization_thread = threading.Thread(target=self._visualization_loop, daemon=True)
        self.visualization_thread.start()
        print("📺 启动测试可视化窗口")

    def _stop_visualization(self):
        """停止可视化线程"""
        if self.visualization_running:
            self.visualization_running = False
            if self.visualization_thread and self.visualization_thread.is_alive():
                self.visualization_thread.join(timeout=1.0)
            print("📺 停止测试可视化窗口")

    def _visualization_loop(self):
        """可视化循环"""
        window_name = "GR00T 可视化测试 - 按 'q' 退出"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1200, 800)

        try:
            while self.visualization_running:
                with self.observation_lock:
                    observation = self.current_observation

                if observation is not None:
                    display_image = self._create_visualization_image(observation)
                    if display_image is not None:
                        cv2.imshow(window_name, display_image)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.visualization_running = False
                    break

                time.sleep(0.01)

        except Exception as e:
            print(f"❌ 可视化线程出错: {e}")
        finally:
            cv2.destroyWindow(window_name)

    def _create_visualization_image(self, observation):
        """创建可视化图像"""
        # 相机配置
        camera_configs = [
            ("video.cam_left_high", "左臂高位相机"),
            ("video.cam_left_wrist", "左臂腕部相机"),
            ("video.cam_right_wrist", "右臂腕部相机")
        ]

        # 获取所有有效的相机图像
        valid_images = []
        valid_names = []

        for cam_key, cam_name in camera_configs:
            if cam_key in observation:
                image_data = observation[cam_key]
                if isinstance(image_data, np.ndarray) and image_data.size > 0:
                    processed_img = self._prepare_image_for_display(image_data)
                    if processed_img is not None:
                        valid_images.append(processed_img)
                        valid_names.append(cam_name)

        if not valid_images:
            return self._create_placeholder_image("等待测试数据...")

        # 创建3x1网格布局
        target_width = 400
        target_height = 300
        grid_width = 3 * target_width
        grid_height = target_height
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        # 调整图像尺寸并填充网格
        for i, (img, name) in enumerate(zip(valid_images, valid_names)):
            if i >= 3:  # 最多显示3个相机
                break

            resized = cv2.resize(img, (target_width, target_height))
            x_start = i * target_width
            x_end = (i + 1) * target_width

            grid_image[0:target_height, x_start:x_end] = resized

            # 添加标签
            cv2.putText(grid_image, name, (x_start + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 添加状态信息
        timestamp = time.strftime("%H:%M:%S")
        cv2.putText(grid_image, f"GR00T可视化测试 - {timestamp}",
                   (10, grid_height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if "annotation.human.action.task_description" in observation:
            task_desc = observation["annotation.human.action.task_description"]
            task_text = f"任务: {task_desc[0] if isinstance(task_desc, list) and task_desc else str(task_desc)}"
            cv2.putText(grid_image, task_text, (10, grid_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return grid_image

    def _prepare_image_for_display(self, image_data):
        """准备图像用于显示"""
        try:
            if not isinstance(image_data, np.ndarray):
                return None

            if image_data.ndim == 4 and image_data.shape[0] == 1:
                img = image_data[0]
            elif image_data.ndim == 3:
                img = image_data
            else:
                return None

            if img.shape[-1] == 4:
                img = img[..., :3]

            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            if img.shape[-1] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            return img
        except Exception as e:
            print(f"❌ 准备图像显示出错: {e}")
            return None

    def _create_placeholder_image(self, message):
        """创建占位符图像"""
        width, height = 1200, 800
        img = np.zeros((height, width, 3), dtype=np.uint8)

        cv2.putText(img, message, (width // 2 - 200, height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        instructions = [
            "GR00T 可视化功能测试",
            "这个窗口显示从仿真环境发送给推理服务的观测数据",
            "按 'q' 键退出测试",
            "测试脚本将模拟观测数据更新"
        ]

        for i, instruction in enumerate(instructions):
            cv2.putText(img, instruction, (50, height - 100 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return img

    def generate_test_observation(self):
        """生成测试观测数据"""
        # 创建彩色测试图像
        def create_test_image(width=640, height=480, pattern="gradient"):
            img = np.zeros((height, width, 3), dtype=np.uint8)

            if pattern == "gradient":
                for i in range(height):
                    for j in range(width):
                        img[i, j, 0] = int(255 * j / width)
                        img[i, j, 1] = int(255 * i / height)
                        img[i, j, 2] = 128
            elif pattern == "chessboard":
                square_size = 50
                for i in range(0, height, square_size):
                    for j in range(0, width, square_size):
                        color = 255 if ((i//square_size + j//square_size) % 2) == 0 else 0
                        img[i:i+square_size, j:j+square_size] = color

            return img[np.newaxis, ...]  # 添加batch维度

        observation = {
            "video.cam_left_high": create_test_image(pattern="gradient"),
            "video.cam_left_wrist": create_test_image(pattern="chessboard"),
            "video.cam_right_wrist": create_test_image(pattern="gradient"),
            "state.left_arm": np.random.rand(1, 7),
            "state.right_arm": np.random.rand(1, 7),
            "state.left_hand": np.random.rand(1, 6),
            "state.right_hand": np.random.rand(1, 6),
            "state.waist": np.random.rand(1, 3),
            "annotation.human.action.task_description": ["测试观测数据可视化"]
        }

        return observation

    def run_test(self):
        """运行测试"""
        print("开始GR00T可视化功能测试...")
        print("按 'q' 键退出测试窗口")

        try:
            for i in range(100):  # 运行100次更新
                # 生成新的测试观测数据
                test_obs = self.generate_test_observation()

                # 更新当前观测数据
                with self.observation_lock:
                    self.current_observation = test_obs

                print(f"更新测试观测数据 #{i+1}")

                # 等待一段时间
                time.sleep(0.5)

                # 检查是否退出
                if not self.visualization_running:
                    break

        except KeyboardInterrupt:
            print("\n测试被用户中断")

        finally:
            self._stop_visualization()
            print("测试完成")

if __name__ == "__main__":
    test = TestVisualization()
    test.run_test()


