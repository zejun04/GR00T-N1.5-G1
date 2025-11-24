# /home/shenlan/GR00T-VLA/g1_xr_teleoperate/unitree_sim_isaaclab/action_provider/action_provider_gr00t.py

import numpy as np
import requests
import json
import time
from typing import Dict, Any, Tuple
import cv2
import sys
import os
import torch
import logging

try:
    import json_numpy
    json_numpy.patch()
    JSON_NUMPY_AVAILABLE = True
    print("✅ json_numpy configured successfully")
except ImportError:
    JSON_NUMPY_AVAILABLE = False
    print("❌ json_numpy not available, using fallback serialization")

class GR00TActionProvider:
    def __init__(self, env, args):
        """
        GR00T动作提供器
        """
        self.env = env
        self.args = args
        self.host = getattr(args, 'gr00t_host', 'localhost')
        self.port = getattr(args, 'gr00t_port', 8000)
        
        # 必需的属性
        self.name = "GR00TActionProvider"
        
        # 设置默认图像尺寸
        self.image_size = (640, 480)  # GR00T期望的默认尺寸(640,480)
        
        # 动作序列相关属性
        self.action_sequence = None
        self.current_step = 0
        self.sequence_length = 16  # GR00T返回的序列长度
        self.last_sequence_time = 0
        self.sequence_request_interval = 2.0  # 每2秒请求新序列
        
        # 检查环境的动作空间
        self._check_action_space()
        
        # 初始化HTTP客户端
        self.session = requests.Session()
        self.base_url = f"http://{self.host}:{self.port}"
        
        # 测试连接
        self._test_connection()
        
        print("✅ GR00T Action Provider initialized successfully")
    
    def _check_action_space(self):
        """检查环境的动作空间"""
        try:
            if hasattr(self.env, 'action_space'):
                print(f"🟡 环境动作空间: {self.env.action_space}")
                print(f"🟡 动作空间形状: {self.env.action_space.shape}")
                print(f"🟡 动作空间类型: {type(self.env.action_space)}")
                
                # 获取动作空间的正确维度
                if hasattr(self.env.action_space, 'shape'):
                    # 动作空间形状是 (1, 43)，我们需要43维的动作
                    if len(self.env.action_space.shape) == 2:
                        self.action_dim = self.env.action_space.shape[1]  # 获取43
                        self.action_shape = (1, self.action_dim)  # 保存完整的形状
                    else:
                        self.action_dim = self.env.action_space.shape[0]
                        self.action_shape = (self.action_dim,)
                    print(f"🟡 动作维度: {self.action_dim}, 动作形状: {self.action_shape}")
                else:
                    print("⚠️ 无法获取动作空间形状，使用默认维度")
                    self.action_dim = 43
                    self.action_shape = (1, 43)
            else:
                print("⚠️ 环境没有action_space属性，使用默认维度")
                self.action_dim = 43
                self.action_shape = (1, 43)
                
        except Exception as e:
            print(f"❌ 检查动作空间时出错: {e}")
            self.action_dim = 43
            self.action_shape = (1, 43)
        
    
    def _test_connection(self):
        """测试与GR00T服务的连接"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Successfully connected to GR00T inference service")
            else:
                print(f"⚠️ GR00T service returned status: {response.status_code}")
        except Exception as e:
            print(f"❌ Failed to connect to GR00T service: {e}")
            print("Please make sure GR00T inference service is running:")
            print("python scripts/inference_service.py --server --http-server --port 8000 --embodiment_tag gr1 --data_config so100")
            raise e
    
    def start(self):
        """启动动作提供器"""
        print("🟢 GR00T Action Provider started")
    
    def stop(self):
        """停止动作提供器"""
        print("🔴 GR00T Action Provider stopped")
    
    def cleanup(self):
        """清理资源"""
        self.close()
    
    def get_action(self, env=None):
        """
        从GR00T服务获取动作
        
        Args:
            env: 仿真环境（为了接口兼容性）
            
        Returns:
            torch.Tensor: 动作张量，符合仿真环境期望的格式
        """
        # 如果传入了新的env，更新当前env
        if env is not None:
            self.env = env
        try:
            current_time = time.time()
            
            # 检查是否需要获取新的动作序列
            if (self.action_sequence is None or 
                self.current_step >= self.sequence_length):
                
                # print("🔄 获取新的动作序列...")
                self.action_sequence = self._get_new_action_sequence()
                self.current_step = 0
                self.last_sequence_time = current_time

            # 从序列中提取当前步骤的动作
            current_action = self._extract_step_action(self.action_sequence, self.current_step)
            self.current_step += 1
            
            # 将动作转换为仿真环境期望的格式
            action_tensor = self._convert_to_env_action(current_action)
            return action_tensor
                
        except Exception as e:
            print(f"❌ 从GR00T获取动作时出错: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_action()
    
    def _get_new_action_sequence(self) -> Dict[str, np.ndarray]:
        """从GR00T服务获取完整的动作序列"""
        try:
            # 准备观测数据
            observation = self.prepare_observation()
            print("观测：",observation)
            
            # 使用json_numpy序列化
            json_data = json_numpy.dumps({"observation": observation})
            headers = {'Content-Type': 'application/json'}
            data = json_data
            # print("data: ",data)
            # 发送请求到GR00T服务
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/act",
                data=data,
                headers=headers,
                timeout=10.0
            )
            
            if response.status_code == 200:
                # 解析响应
                try:
                    action_data = json_numpy.loads(response.text)
                    # print("使用json_numpy解析响应成功")
                except Exception as e:
                        print(f"❌ 使用json_numpy解析响应失败: {e}")
                
                inference_time = time.time() - start_time
                print(f"✅ GR00T推理成功 - 时间: {inference_time:.3f}s")
                
                # 验证动作序列结构
                # print("动作是：",action_data)
                return action_data
            else:
                print(f"❌ GR00T服务返回错误: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"❌ 获取动作序列失败: {e}")
    
    
    def _extract_step_action(self, action_sequence: Dict[str, np.ndarray], step_idx: int) -> Dict[str, np.ndarray]:
        """
        从动作序列中提取指定步骤的动作
        
        Args:
            action_sequence: 完整的动作序列
            step_idx: 要提取的步骤索引
            
        Returns:
            Dict[str, np.ndarray]: 当前步骤的动作
        """
        current_action = {}
        
        for key, sequence in action_sequence.items():
            if isinstance(sequence, np.ndarray) and len(sequence.shape) == 2:
                # 确保索引在有效范围内
                if step_idx < sequence.shape[0]:
                    current_action[key] = sequence[step_idx]
                else:
                    # 如果超出范围，使用最后一步
                    current_action[key] = sequence[-1]
                    print(f"⚠️ 步骤索引 {step_idx} 超出范围，使用最后一步")
            else:
                current_action[key] = sequence
        print("动作是：",current_action)
        return current_action
    
    
    def _convert_to_env_action(self, action_data: Dict[str, np.ndarray]) -> torch.Tensor:
        """
        将GR00T返回的动作字典转换为仿真环境期望的torch.Tensor格式
        
        Args:
            action_data: GR00T返回的动作字典
            
        Returns:
            torch.Tensor: 仿真环境期望的动作张量
        """
        try:
            # 将动作字典转换为完整43维动作向量
            action_vector = self._build_full_action_vector(action_data)
            
            
            # 确保动作维度
            action_vector = self._ensure_action_dimension(action_vector)
            
            # 重塑为环境期望的形状
            action_vector = action_vector.reshape(self.action_shape)
            
            # 转换为torch张量
            action_tensor = torch.from_numpy(action_vector).to(self.env.device)
            
            return action_tensor
            
        except Exception as e:
            print(f"❌ 转换动作格式时出错: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_action()
        
    
    def _build_full_action_vector(self, action_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        构建完整的43维动作向量
        
        根据G129机器人的关节顺序：
        - 身体关节：29个（腿部12 + 腰部3 + 手臂14）
        - 手部关节：14个
        """
        try:
            # 创建43维的零向量
            full_action = np.zeros(43, dtype=np.float32)
            
            # 映射GR00T动作到完整的关节空间[左闭，右开)
            # action.left/right_arm/hand is provided by GR00T
            # index,example 15-21 is used for isaac_sim
            action_mappings = [
                ('action.left_arm', 15, 22),    # 左臂 -> 索引15-21
                ('action.left_hand', 29, 36),   # 右臂 -> 索引22-28
                ('action.right_arm', 22, 29),   # 左手 -> 索引29-35
                ('action.right_hand', 36, 43)   # 右手 -> 索引36-42
            ]
            
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 0

            used_indices = 0
            for key, start_idx, end_idx in action_mappings:
                if key in action_dict:
                    action_part = action_dict[key]
                    dim = end_idx - start_idx
                    
                    if action_part.shape[0] == dim:
                        
                        full_action[start_idx:end_idx] = action_part
                        used_indices += dim
                        # if self._debug_count % 10 == 0:
                        #     print(f"✅ {key}:")
                        #     print(f"   动作范围: [{action_part.min():.4f}, {action_part.max():.4f}]")
                    else:
                        print(f"⚠️ 动作部分 {key} 维度不匹配: {action_part.shape[0]} != {dim}")
                        # 使用零向量替代
                        full_action[start_idx:end_idx] = np.zeros(dim, dtype=np.float32)
                else:
                    print(f"⚠️ 缺少动作部分: {key}")
                    # 使用零向量填充
                    full_action[start_idx:end_idx] = np.zeros(dim, dtype=np.float32)
            
            
            # 对于未映射的部分（腿部、腰部等），保持为零
            # 这些部分将由仿真环境处理或保持默认位置
            print("G1_action",full_action)
            return full_action
            
        except Exception as e:
            print(f"❌ 构建完整动作向量时出错: {e}")
            # 返回零向量
            return np.zeros(43, dtype=np.float32)
    
    def _ensure_action_dimension(self, action_vector: np.ndarray) -> np.ndarray:
        """
        确保动作向量与环境的动作空间维度匹配
        """
        current_dim = action_vector.shape[0]
        
        if current_dim == self.action_dim:
            # 维度匹配，直接返回
            return action_vector
        elif current_dim > self.action_dim:
            # 动作向量维度太大，截断
            print(f"⚠️ 动作向量维度过大 ({current_dim} > {self.action_dim})，进行截断")
            return action_vector[:self.action_dim]
        else:
            # 动作向量维度太小，填充零
            print(f"⚠️ 动作向量维度过小 ({current_dim} < {self.action_dim})，进行零填充")
            padded_action = np.zeros(self.action_dim, dtype=np.float32)
            padded_action[:current_dim] = action_vector
            return padded_action
    
    def prepare_observation(self) -> Dict[str, Any]:
        """
        准备GR00T模型所需的观测数据
        
        Returns:
            Dict: 符合GR00T输入格式的观测数据
        """
        try:
            # 获取相机图像
            camera_obs = self._get_camera_observations()
            
            # 获取机器人状态
            robot_state = self._get_robot_state()
            
            # 构建GR00T观测字典(whole body)
            # observation = {
            #     "video.rs_view": camera_obs["rs_view"],
            #     "state.left_leg": robot_state["left_leg"],
            #     "state.right_leg": robot_state["right_leg"],
            #     "state.waist": robot_state["waist"],
            #     "state.left_arm": robot_state["left_arm"],
            #     "state.right_arm": robot_state["right_arm"], 
            #     "state.left_hand": robot_state["left_hand"],
            #     "state.right_hand": robot_state["right_hand"],
            #     "annotation.human.action.task_description": ["Pick up the red apple and put it on the plate"]
            # }

            observation = {
                "video.rs_view": camera_obs["rs_view"],
                "state.left_arm": robot_state["left_arm"],
                "state.left_hand": robot_state["left_hand"],
                "state.right_arm": robot_state["right_arm"], 
                "state.right_hand": robot_state["right_hand"],
                "annotation.human.action.task_description": ["Pick up the red apple and put it on the plate"]
            }
            
            print(f"指令：{observation['annotation.human.action.task_description']}")
            #print("观测是：", observation)
            return observation
            
        except Exception as e:
            print(f"❌ 准备观测数据时出错: {e}")
            return self._get_default_observation()
    
    def _get_camera_observations(self) -> Dict[str, np.ndarray]:
        """
        从仿真环境获取相机图像并调整到GR00T期望的尺寸
        """
        # if hasattr(self, '_debug_count'):
        #     self._debug_count += 1
        # else:
        #     self._debug_count = 0
        # try:
        #     camera_data = {}
        #     camera_image = None
        #     target_cam_name = 'front_camera' 
            
        #     # 直接从环境场景传感器(Scene Sensors)获取 (Isaac Lab 标准方式)
        #     if hasattr(self.env, 'scene') and hasattr(self.env.scene, 'sensors'):
        #         if target_cam_name in self.env.scene.sensors:
        #             sensor = self.env.scene.sensors[target_cam_name]
        #             if hasattr(sensor, 'data') and hasattr(sensor.data, 'output'):
        #                 if 'rgb' in sensor.data.output:
        #                     image_tensor = sensor.data.output['rgb']
                            
        #                     if isinstance(image_tensor, torch.Tensor):
        #                         camera_image = image_tensor.clone().detach().cpu().numpy()
        #                     else:
        #                         camera_image = image_tensor
                                
        #                     # print(f"📷 获取到 {target_cam_name}: {camera_image.shape}")

        #     opencv_image = camera_image.squeeze(axis=0)
        #     cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR, opencv_image)
        #     time_stamp = time.strftime('%Y%m%d_%H%M%S') + f'{time.time()%1:.3f}'[1:]
        #     save_dir = os.path.join(os.getcwd(), 'rs_img/')      # 当前终端目录/rs_img
        #     os.makedirs(save_dir, exist_ok=True)                # 没有就自动建

            
        #     file_name = os.path.join(save_dir, f'{self._debug_count}.jpg')
        #     cv2.imwrite(file_name, opencv_image)
        #     if self._debug_count % 10 == 0: 
        #         print('rs_view 已写入', file_name)

        #     # 处理获取到的图像
        #     if camera_image is not None:
        #         # 如果是 RGBA (4通道)，去掉 Alpha 通道转为 RGB
        #         if camera_image.shape[-1] == 4:
        #             camera_image = camera_image[..., :3]
        #         # print("camera shape:", camera_image.shape)
        #         processed_image = self._process_camera_image(camera_image)
        #         camera_data["rs_view"] = processed_image
        #         return camera_data
                
        #     else:
        #         print(f"⚠️ 未找到相机数据: {target_cam_name}")
                
        # except Exception as e:
        #     logging.error(f"❌ 获取相机数据时出错: {e}")
        #     import traceback
        #     traceback.print_exc()
        
        # 如果以上方法都失败，返回测试数据
        print("❌ 使用测试相机数据!")
        return {"rs_view": self._get_test_camera_data()}
    
    def _process_camera_image(self, image: np.ndarray) -> np.ndarray:
        """
        处理相机图像：调整尺寸为GR00T期望的尺寸，确保格式正确
        
        Args:
            image: 原始相机图像
            
        Returns:
            np.ndarray: 处理后的图像
        """
        target_width, target_height = self.image_size
        
        # 确保图像是NHWC格式 (Batch, Height, Width, Channels)
        if len(image.shape) == 4:
            if image.shape[1] == 3:  # NCHW格式
                # 转换为NHWC
                image = image.transpose(0, 2, 3, 1)
        elif len(image.shape) == 3:
            # 如果是HWC，添加batch维度
            image = image[np.newaxis, ...]
        
        # 调整图像尺寸到目标尺寸
        batch_size = image.shape[0]
        resized_images = np.zeros((batch_size, target_height, target_width, 3), dtype=np.uint8)
        
        for i in range(batch_size):
            img = image[i]
            
            # 确保图像是uint8类型
            if img.dtype != np.uint8:
                if img.max() <= 1.0:  # 假设是0-1范围的float
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # 调整尺寸到目标尺寸
            resized_img = cv2.resize(img, (target_width, target_height))
            resized_images[i] = resized_img
        
        return resized_images
    
    def _get_robot_state(self) -> Dict[str, np.ndarray]:
        """
        从 Isaac Lab 的 Articulation 获取真实的机器人关节状态
        """
        try:
            # 获取机器人 Articulation 对象
            robot_articulation = None
            if hasattr(self.env, 'robot'):
                robot_articulation = self.env.robot
            elif hasattr(self.env.scene, 'robot'):
                robot_articulation = self.env.scene['robot']
            elif hasattr(self.env.scene, 'articulations') and len(self.env.scene.articulations) > 0:
                robot_articulation = list(self.env.scene.articulations.values())[0]
            
            if robot_articulation is None:
                raise ValueError("未在环境中找到机器人 Articulation 对象")
            
            # 获取关节位置数据
            joint_pos_tensor = robot_articulation.data.joint_pos[0]
            
            if isinstance(joint_pos_tensor, torch.Tensor):
                joint_pos = joint_pos_tensor.detach().cpu().numpy()
            else:
                joint_pos = joint_pos_tensor
            
            joint_pos = joint_pos.astype(np.float32)
            # print("获取到的关节位置",joint_pos)
            # 根据 G129 的 43 维动作空间映射关节索引
            state_data = {}

            # debug专用
            # joint_pos[15:22] = 1.0
            # joint_pos[22:29] = 2.0
            # joint_pos[29:36] = 3.0
            # joint_pos[36:43] = 4.0

            state_data["left_leg"] = joint_pos[0:6].reshape(1, 6)
            state_data["right_leg"] = joint_pos[6:12].reshape(1, 6)
            # 腰部: 索引 12-14 (3个关节)
            state_data["waist"] = joint_pos[12:15].reshape(1, 3)
            # 左臂: 索引 15-21 (7个关节)
            state_data["left_arm"] = joint_pos[15:22].reshape(1, 7)
            
            # 右臂: 索引 22-28 (7个关节)
            state_data["right_arm"] = joint_pos[22:29].reshape(1, 7)
            
            # 左手: 索引 29-35 (7个关节)
            state_data["left_hand"] = joint_pos[29:36].reshape(1, 7)
            
            # 右手: 索引 36-42 (7个关节)
            state_data["right_hand"] = joint_pos[36:43].reshape(1, 7)
            
            
            
            # Debug 打印
            if hasattr(self, '_debug_count'):
                self._debug_count += 1
            else:
                self._debug_count = 0
            
            if self._debug_count % 10 == 0:
                print(f"🔍 机器人状态已更新:")
                for key, value in state_data.items():
                    print(f"   {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            
            return state_data
        
        except Exception as e:
            print(f"❌ 从 Isaac Lab 获取机器人状态时出错: {e}")
            import traceback
            traceback.print_exc()
    
    
    def reset(self):
        """重置动作提供器状态"""
        print("🔄 GR00T Action Provider reset")
        self.action_sequence = None
        self.current_step = 0
    
    def close(self):
        """关闭连接"""
        self.session.close()
        print("🔒 GR00T Action Provider closed")

    def _get_test_camera_data(self) -> np.ndarray:
        """
        从外部视频文件读取图像作为测试相机数据
        
        Returns:
            np.ndarray: 处理后的视频帧，形状为 (1, 480, 640, 3)
        """
        try:
            video_path = None
            if video_path is None:
                # 如果没有配置视频路径，使用默认测试视频
                default_video = "/home/shenlan/GR00T-VLA/Isaac-GR00T/datasets/g1-pick-apple/videos/chunk-000/observation.images.ego_view/episode_000000.mp4"
                if os.path.exists(default_video):
                    video_path = default_video
                else:
                    # 生成彩色测试图像作为备选
                    print("⚠️ 未找到测试视频文件，使用生成的测试图像")
                    return self._generate_test_image()
            
            # 初始化视频捕获（作为实例变量，避免重复创建）
            if not hasattr(self, '_test_video_cap'):
                self._test_video_cap = cv2.VideoCapture(video_path)
                self._test_video_frame_count = 0
                
                if not self._test_video_cap.isOpened():
                    print(f"❌ 无法打开测试视频文件: {video_path}")
                    return self._generate_test_image()
                else:
                    print(f"✅ 成功加载测试视频: {video_path}")
            
            # 读取视频帧
            ret, frame = self._test_video_cap.read()
            
            if not ret:
                # 视频结束，重置到开头
                self._test_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._test_video_cap.read()
                
                if not ret:
                    print("❌ 无法从视频读取帧，使用测试图像")
                    return self._generate_test_image()
            
            # 转换颜色空间 BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 调整尺寸到目标尺寸 (640, 480)
            target_width, target_height = self.image_size
            resized_frame = cv2.resize(frame_rgb, (target_width, target_height))
            
            # 添加batch维度 (1, H, W, C)
            final_frame = resized_frame[np.newaxis, ...]
            
            # 帧计数和调试信息
            self._test_video_frame_count += 1
            if self._test_video_frame_count % 30 == 0:  # 每30帧打印一次
                print(f"📹 测试视频帧: {self._test_video_frame_count}, 形状: {final_frame.shape}")
            
            return final_frame
            
        except Exception as e:
            print(f"❌ 读取测试视频时出错: {e}")
            import traceback
            traceback.print_exc()