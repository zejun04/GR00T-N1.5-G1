import os
import subprocess
from pathlib import Path

def convert_av1_to_h264(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for mp4_file in input_path.rglob("*.mp4"):
        try:
            # 检查是否为 AV1 编码
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name', '-of', 'default=noprint_wrappers=1:nokey=1',
                str(mp4_file)
            ], capture_output=True, text=True)
            
            codec = result.stdout.strip()
            print(f"文件 {mp4_file.name} 的编码格式: {codec}")
            
            relative_path = mp4_file.relative_to(input_path)
            output_file = output_path / relative_path
            
            # 创建输出目录
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查输出文件是否已存在
            if output_file.exists():
                print(f"输出文件已存在，跳过: {output_file}")
                successful += 1
                continue
            
            # 转换为 H.264，添加更详细的日志
            print(f"开始转换: {mp4_file.name}")
            process = subprocess.run([
                'ffmpeg', '-i', str(mp4_file),
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '23',
                '-c:a', 'aac', '-b:a', '128k',  # 添加音频编码
                '-y', str(output_file)
            ], capture_output=True, text=True, timeout=300)  # 添加5分钟超时
            
            if process.returncode == 0:
                print(f"✓ 成功转换: {mp4_file.name}")
                successful += 1
            else:
                print(f"✗ 转换失败: {mp4_file.name}")
                print(f"错误信息: {process.stderr}")
                failed += 1
                
        except subprocess.TimeoutExpired:
            print(f"✗ 转换超时: {mp4_file.name}")
            failed += 1
        except Exception as e:
            print(f"✗ 处理文件时出错 {mp4_file.name}: {e}")
            failed += 1
    
    print(f"\n转换完成: 成功 {successful}, 失败 {failed}")

# 使用示例
convert_av1_to_h264(
    "/home/shenlan/GR00T-VLA/Isaac-GR00T/datasets/G1_Dex3_BlockStacking_Dataset/videos/chunk-000/observation.images.cam_right_wrist/AV1",
    "/home/shenlan/GR00T-VLA/Isaac-GR00T/datasets/G1_Dex3_BlockStacking_Dataset/videos/chunk-000/observation.images.cam_right_wrist"
)
