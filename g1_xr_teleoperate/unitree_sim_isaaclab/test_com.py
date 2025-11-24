# test_com.py

import sys
import os
import numpy as np

def test_step_by_step():
    """逐步测试GR00T集成"""
    
    print("🧪 开始逐步测试GR00T集成...")
    
    
    # 步骤1: 使用AppLauncher初始化Isaac Sim环境
    print("\n1. 初始化Isaac Sim环境...")
    try:
        from isaaclab.app import AppLauncher
        
        # 创建简单的参数类
        class Args:
            device = "cpu"
            task = "Isaac-Play-G129-Dex1-Joint"
            headless = False  # 使用headless模式避免图形界面
        
        args = Args()
        
        # 使用AppLauncher初始化Isaac Sim环境
        app_launcher = AppLauncher(headless=args.headless)
        simulation_app = app_launcher.app
        
        print("✅ AppLauncher初始化成功")
        
        # 现在可以安全导入Isaac Sim相关模块
        import gymnasium as gym
        from tasks.utils.parse_cfg import parse_env_cfg
        
        # 解析环境配置
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
        
        # 设置资产路径（如果配置允许）
        if hasattr(env_cfg, 'scene'):
            if hasattr(env_cfg.scene, 'assets'):
                env_cfg.scene.assets.asset_root = isaaclab_assets_path
                print(f"✅ 设置资产根路径: {isaaclab_assets_path}")
        
        env = gym.make(args.task, cfg=env_cfg)
        
        print("✅ 环境创建成功")
        
        # 步骤2: 测试GR00T动作提供器初始化
        print("\n2. 测试GR00T动作提供器初始化...")
        try:
            from action_provider.action_provider_gr00t import GR00TActionProvider
            
            action_provider = GR00TActionProvider(env, args)
            print("✅ GR00T动作提供器初始化成功")
            
            # 步骤3: 测试观测数据准备
            print("\n3. 测试观测数据准备...")
            try:
                observation = action_provider.prepare_observation()
                print("✅ 观测数据准备成功")
                for key, value in observation.items():
                    if isinstance(value, np.ndarray):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: {value}")
                
                # 步骤4: 测试动作获取
                print("\n4. 测试动作获取...")
                try:
                    action = action_provider.get_action()
                    print("✅ 动作获取成功")
                    for key, value in action.items():
                        if isinstance(value, np.ndarray):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                        else:
                            print(f"  {key}: {value}")
                    
                    print("\n🎉 所有测试通过！GR00T集成正常工作")
                    
                    # 清理资源
                    action_provider.close()
                    env.close()
                    
                except Exception as e:
                    print(f"❌ 动作获取失败: {e}")
                    import traceback
                    traceback.print_exc()
                    
            except Exception as e:
                print(f"❌ 观测数据准备失败: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"❌ GR00T动作提供器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            
    except Exception as e:
        print(f"❌ 环境创建失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 确保关闭应用
        try:
            simulation_app.close()
            print("🔒 应用已关闭")
        except:
            pass

if __name__ == "__main__":
    # 添加项目路径
    project_root = "/home/shenlan/GR00T-VLA/g1_xr_teleoperate/unitree_sim_isaaclab"
    sys.path.append(project_root)
    
    test_step_by_step()