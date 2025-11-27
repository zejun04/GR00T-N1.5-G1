#!/usr/bin/env python3
"""
直接发送关节控制命令测试DDS通信
"""

import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC

def test_joint_control():
    """测试关节控制"""
    print("初始化DDS通信...")
    ChannelFactoryInitialize(1)  # 使用信道1
    
    # 创建发布者
    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()
    
    # 创建订阅者
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init()
    
    print("DDS通信初始化完成")
    
    # 等待订阅者连接
    time.sleep(1)
    
    # 创建控制命令 - 使用正确的构造函数
    lowcmd = LowCmd_(
        mode_pr=0,  # 保护模式
        mode_machine=0,  # 机器模式
        motor_cmd=[unitree_hg_msg_dds__LowCmd_() for _ in range(35)],  # 35个电机命令
        reserve=0,  # 保留字段
        crc=0  # CRC校验
    )
    
    # 设置关节控制模式
    for i in range(35):
        lowcmd.motor_cmd[i].mode = 10  # 位置控制模式
        lowcmd.motor_cmd[i].q = 0.0    # 目标位置
        lowcmd.motor_cmd[i].dq = 0.0   # 目标速度
        lowcmd.motor_cmd[i].kp = 100.0  # 位置增益
        lowcmd.motor_cmd[i].kd = 1.0    # 速度增益
        lowcmd.motor_cmd[i].tau = 0.0   # 力矩
    
    # 测试控制几个关键关节
    test_joints = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]  # 手臂关节
    
    print("开始关节控制测试...")
    
    for i in range(5):  # 测试5个周期
        print(f"\n测试周期 {i+1}")
        
        # 设置不同的关节位置
        for joint_idx in test_joints:
            # 设置目标位置（小幅摆动）
            target_pos = np.sin(i * 0.5) * 0.1  # -0.1到0.1弧度
            lowcmd.motor_cmd[joint_idx].q = target_pos
            
            print(f"关节 {joint_idx}: 目标位置 {target_pos:.3f} rad")
        
        # 发送控制命令
        publisher.Write(lowcmd)
        print("控制命令已发送")
        
        # 读取状态（非阻塞方式）
        state_msg = LowState_()
        if subscriber.Read(state_msg, 0):
            print("收到机器人状态反馈")
            # 显示几个关键关节的实际位置
            for joint_idx in test_joints[:3]:  # 只显示前3个关节
                actual_pos = state_msg.motor_state[joint_idx].q
                print(f"关节 {joint_idx} 实际位置: {actual_pos:.3f} rad")
        else:
            print("未收到状态反馈")
        
        time.sleep(2)  # 等待2秒
    
    print("\n测试完成")

if __name__ == "__main__":
    test_joint_control()