#!/usr/bin/env python3
"""
WebSocket客户端测试脚本
用于测试WebSocket服务器是否正常接收和发送数据
"""

import asyncio
import websockets
import json
import ssl
import time
import numpy as np
import msgpack

def create_test_hand_data():
    """创建模拟的手势数据"""
    # 模拟25个手部关节点数据，每个关节点16个元素
    hand_data = []
    for i in range(25):
        # 创建一个4x4变换矩阵
        matrix = np.eye(4)
        # 添加一些随机位置偏移
        matrix[0, 3] = i * 0.01  # x位置
        matrix[1, 3] = i * 0.02  # y位置  
        matrix[2, 3] = i * 0.03  # z位置
        
        # 将矩阵展平为16个元素的列表
        hand_data.extend(matrix.flatten())
    
    return hand_data

async def test_websocket_connection():
    """测试WebSocket连接和数据传输"""
    
    # WebSocket服务器地址
    uri = "wss://192.168.1.51:8012"
    
    # 创建SSL上下文（忽略证书验证）
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    try:
        print(f"尝试连接到: {uri}")
        
        async with websockets.connect(uri, ssl=ssl_context) as websocket:
            print("✓ WebSocket连接成功建立")
            
            # 发送测试消息
            test_message = {
                "type": "test",
                "message": "Hello from test client",
                "timestamp": time.time()
            }
            
            # 使用MessagePack格式发送
            await websocket.send(msgpack.packb(test_message))
            print("✓ 测试消息已发送")
            
            # 等待服务器响应
            print("等待服务器响应...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                print(f"✓ 收到服务器响应: {response}")
            except asyncio.TimeoutError:
                print("⚠ 服务器响应超时（可能正常，服务器可能不主动发送数据）")
            
            # 测试发送手势数据
            print("\n测试发送手势数据...")
            
            # 创建模拟手势数据
            left_hand_data = create_test_hand_data()
            right_hand_data = create_test_hand_data()
            
            hand_message = {
                "type": "HAND_MOVE",
                "value": {
                    "left": left_hand_data,
                    "right": right_hand_data,
                    "leftState": {
                        "pinch": True,
                        "pinchValue": 0.8,
                        "squeeze": False,
                        "squeezeValue": 0.2
                    },
                    "rightState": {
                        "pinch": False,
                        "pinchValue": 0.3,
                        "squeeze": True,
                        "squeezeValue": 0.9
                    }
                },
                "timestamp": time.time()
            }
            
            await websocket.send(msgpack.packb(hand_message))
            print("✓ 手势数据已发送")
            
            # 保持连接一段时间，观察是否有数据回传
            print("\n保持连接10秒，观察数据流...")
            start_time = time.time()
            message_count = 0
            
            while time.time() - start_time < 10:
                try:
                    # 设置较短的超时时间
                    message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                    message_count += 1
                    print(f"收到消息 #{message_count}: {message[:100]}...")  # 只显示前100字符
                except asyncio.TimeoutError:
                    # 超时是正常的，继续等待
                    pass
                
                # 每秒发送一次心跳
                if int(time.time() - start_time) > int(time.time() - start_time - 1):
                    heartbeat = {
                        "type": "heartbeat",
                        "timestamp": time.time()
                    }
                    await websocket.send(msgpack.packb(heartbeat))
            
            print(f"\n测试完成。共收到 {message_count} 条消息")
            
    except websockets.exceptions.InvalidURI:
        print("✗ 无效的URI格式")
    except websockets.exceptions.InvalidHandshake:
        print("✗ 握手失败 - 检查服务器是否运行")
    except ConnectionRefusedError:
        print("✗ 连接被拒绝 - 检查服务器IP和端口")
    except Exception as e:
        print(f"✗ 连接失败: {e}")

def test_port_connectivity():
    """测试端口连通性"""
    import socket
    
    print("\n=== 端口连通性测试 ===")
    
    host = "192.168.1.51"
    port = 8012
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(3)
        result = sock.connect_ex((host, port))
        
        if result == 0:
            print(f"✓ 端口 {port} 在 {host} 上已打开")
        else:
            print(f"✗ 端口 {port} 在 {host} 上未响应 (错误代码: {result})")
        
        sock.close()
    except Exception as e:
        print(f"✗ 端口测试失败: {e}")

if __name__ == "__main__":
    print("WebSocket客户端测试工具")
    print("=" * 50)
    
    # 测试端口连通性
    test_port_connectivity()
    
    # 测试WebSocket连接
    print("\n=== WebSocket连接测试 ===")
    asyncio.run(test_websocket_connection())
    
    print("\n测试完成！")