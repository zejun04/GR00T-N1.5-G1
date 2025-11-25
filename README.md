# 一、部署与安装

## 1.1 GR00T的部署测试
创建groot conda 环境
GR00T N1.5的部署参考https://github.com/NVIDIA/Isaac-GR00T/tree/main
适用版本：cuda 12.4(当前使用) 或 cuda 11.8
python=3.10 torch=2.5.0 TorchCodec>=0.1.0


下载NVIDA官方模型：[GR00T-N1.5](https://huggingface.co/nvidia/GR00T-N1.5-3B/tree/main)
里面有三个模型都要全部下载下来，移动到isaac-GR00T/nvidia

部署完成之后：
使用如下命令进行测试：
```
# 服务端
python scripts/inference_service.py --model-path nvidia/GR00T-N1.5-3B --server
# 客户端(另外一个终端)
python scripts/inference_service.py  --client
```


## 1.2 unitree_sim_isaaclab的部署
创建unitree_sim_env conda 环境
确保电脑已经安装isaac sim >=4.5,isaac lab,unitree_sdk2_python
部署[unitree_sim_isaaclab](https://github.com/unitreerobotics/unitree_sim_isaaclab/blob/main/README_zh-CN.md)



其中 --action_source groot 主要添加了`unitree_sim_isaaclab/action_provider/action_provider_gr00t` 修改了`unitree_sim_isaaclab/sim_main.py`

# 二、 启动推理与仿真

将new_embodiment的metadata.json(存于Isaac-GR00T目录下)移动到~/.cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots/869830fc749c35f34771aa5209f923ac57e4564e(重命名为metadata.json)

在isaac-GR00T目录下激活groot环境，运行如下命令启动`推理服务端`(由于是在同一机器上运行，故使用hhtp进行通信)：
```
python scripts/inference_service.py --server --model_path nvidia/GR00T-N1.5-3B --http-server --port 8000 --embodiment_tag new_embodiment --data_config unitree_g1
```



在g1_xr_teleoperate/unitree_sim_isaaclab目录下，激活unitree_sim_env,运行`客户端`(采用三指灵巧手):
```
python sim_main.py --device cuda:0 --enable_cameras --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint --action_source gr00t --robot_type g129 --enable_dex3_dds 
```

# 三、微调训练
# 微调数据集
[G1抓水果 NVIDA](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Teleop-G1/tree/main)
——输入的观测： 
```
"video.rs_view": camera_obs["rs_view"], # 640*480
"state.left_arm": robot_state["left_arm"],
"state.right_arm": robot_state["right_arm"], 
"state.left_hand": robot_state["left_hand"],
"state.right_hand": robot_state["right_hand"],
"state.waist": robot_state["waist"],
"annotation.human.action.task_description": ["Pick up cylinder and  put it on the plane."]
```
——输出：28维度
双手(14) + 双臂(14)
左臂引索[15,21]、右臂引索[22,28]、左手[29,35]、右手[36,42]




[G1堆叠放置物块 宇树](https://huggingface.co/datasets/unitreerobotics/G1_Dex3_BlockStacking_Dataset/tree/main)


# 四、其他步骤文档说明位于/doc