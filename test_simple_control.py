#!/usr/bin/env python3
"""
简单测试DDS通信 - 使用遥操作程序相同的主题和消息格式
"""

import time
import numpy as np
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_

# 使用遥操作程序中的主题定义
kTopicLowCommand_Debug = "rt/lowcmd"
kTopicLowState = "rt/lowstate"

def test_simple_control():
    """简单控制测试"""
    print("初始化DDS通信...")
    ChannelFactoryInitialize(1)  # 使用信道1
    
    # 创建发布者
    publisher = ChannelPublisher(kTopicLowCommand_Debug, LowCmd_)
    publisher.Init()
    
    # 创建订阅者
    subscriber = ChannelSubscriber(kTopicLowState, LowState_)
    subscriber.Init()
    
    print("DDS通信初始化完成")
    
    # 等待订阅者连接
    time.sleep(1)
    
    # 创建控制命令 - 使用遥操作程序中的格式
    lowcmd = LowCmd_()
    
    # 初始化电机命令数组
    lowcmd.motor_cmd = [unitree_hg_msg_dds__LowCmd_() for _ in range(35)]
    
    # 设置默认参数
    lowcmd.mode_pr = 0
    lowcmd.mode_machine = 0
    lowcmd.reserve = 0
    lowcmd.crc = 0
    
    # 设置关节控制模式（使用遥操作程序中的模式）
    for i in range(35):
        lowcmd.motor_cmd[i].mode = 10  # 位置控制模式
        lowcmd.motor_cmd[i].q = 0.0    # 目标位置
        lowcmd.motor_cmd[i].dq = 0.0   # 目标速度
        lowcmd.motor_cmd[i].kp = 100.0  # 位置增益
        lowcmd.motor_cmd[i].kd = 1.0    # 速度增益
        lowcmd.motor_cmd[i].tau = 0.0   # 力矩
    
    # 测试控制手臂关节
    arm_joints = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    
    print("开始简单控制测试...")
    
    for i in range(3):  # 测试3个周期
        print(f"\n测试周期 {i+1}")
        
        # 设置手臂关节位置
        for joint_idx in arm_joints:
            # 小幅摆动
            target_pos = np.sin(i * 0.5) * 0.05  # -0.05到0.05弧度
            lowcmd.motor_cmd[joint_idx].q = target_pos
            
            print(f"关节 {joint_idx}: 目标位置 {target_pos:.3f} rad")
        
        try:
            # 发送控制命令
            publisher.Write(lowcmd)
            print("控制命令已发送")
        except Exception as e:
            print(f"发送命令失败: {e}")
            continue
        
        # 尝试读取状态
        time.sleep(0.5)  # 等待机器人响应
        
        state_msg = LowState_()
        if subscriber.Read(state_msg, 0):  # 非阻塞读取
            print("收到机器人状态反馈")
            # 显示关节位置
            for joint_idx in arm_joints[:3]:  # 只显示前3个关节
                actual_pos = state_msg.motor_state[joint_idx].q
                print(f"关节 {joint_idx} 实际位置: {actual_pos:.3f} rad")
        else:
            print("未收到状态反馈")
        
        time.sleep(1)  # 等待1秒
    
    print("\n测试完成")

if __name__ == "__main__":
    test_simple_control()