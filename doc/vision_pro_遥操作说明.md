Vision Pro   G1遥操作BUG说明

xr_teleoperate 版本：v1.0
# 仿真：

## 1.一直[Dex3_1_Controller] Waiting   to subscribe dds...   
原因：DDS信道不一致
需要确保Isaac lab中的DDS信道与xr_teleoperate保持一致

如：
在isaac lab中的DDS信道在unitree_sim_isaaclab/dds/dds_master.py中定义为“1”
ChannelFactoryInitialize(1)

在xr_teleoperate中的信道是0（xr_teleoperate/teleop/robot_control/robot_arm.py）
ChannelFactoryInitialize(0)  修改为1




## 2.通过https://192.168.1.51:8012/?ws=wss://192.168.1.51:8012(WebSocket)无法访问仿真的摄像头图像

修改相机尺寸符合仿真的大小(640*480)
xr_teleoperate/teleop/teleop_hand_and_arm.py
```
# image client: img_config should be the same as the configuration in image_server.py (of Robot's development computing unit)
    if args.sim:
        img_config = {
            'fps': 30,
            'head_camera_type': 'opencv',
            'head_camera_image_shape': [480, 640],  # Head camera resolution - 仿真环境front_camera是640x480
            'head_camera_id_numbers': [0],
            'wrist_camera_type': 'opencv',
            'wrist_camera_image_shape': [480, 640],  # Wrist camera resolution
            'wrist_camera_id_numbers': [2, 4],
        }

```

修改ip地址为本机ip地址
/home/shenlan/GR00T-VLA/g1_xr_teleoperate/xr_teleoperate/teleop/image_server/image_client.py
```
client = ImageClient(image_show = True, server_address='192.168.1.51', Unit_Test=False) # deployment test
```


启动仿真
cd /home/shenlan/GR00T-VLA/g1_xr_teleoperate/unitree_sim_isaaclab && conda activate unitree_sim_env && python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-Cylinder-G129-Dex3-Joint --enable_dex3_dds --robot_type g129


启动图像传输
cd /home/shenlan/GR00T-VLA/g1_xr_teleoperate/unitree_sim_isaaclab && conda activate unitree_sim_env && python image_server/image_server.py

启动xr_teleoperate
cd /home/shenlan/GR00T-VLA/g1_xr_teleoperate/xr_teleoperate/teleop && conda activate g1_tv && python teleop_hand_and_arm.py --xr-mode=hand --arm=G1_29 --ee=dex3 --sim --record


ps：若出现本地可以访问，但远程无法访问图像的情况
需要关闭VPN


## 进入vision pro无法识别手势(图像接收已正常)




