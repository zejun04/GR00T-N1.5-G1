# GR00T 微调脚本修复文档

## 概述

本文档记录了对 GR00T 微调脚本在处理 pickcube 数据集时遇到的多个关键问题的修复过程。这些修复确保了训练能够正常进行。

## 修复的问题

### 1. IndexError in Statistics Processing

**问题描述：**
```python
IndexError: index 1 is out of bounds for axis 0 with size 1
```

**原因：**
在 `_get_metadata` 方法中，代码尝试访问统计数组的索引，但某些统计值是标量，只有1个元素。

**修复位置：** `gr00t/data/dataset.py` 第 380-395 行

**修复方案：**
```python
# 添加维度检查
max_index = state_action_meta.end - 1
if len(stat) > max_index:
    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()
else:
    if len(stat) == 1:
        dataset_statistics[our_modality][subkey][stat_name] = stat.tolist()
    else:
        continue
```

### 2. FileNotFoundError for JSONL Files

**问题描述：**
```python
FileNotFoundError: [Errno 2] No such file or directory: 'episodes.jsonl'
FileNotFoundError: [Errno 2] No such file or directory: 'tasks.jsonl'
```

**原因：**
pickcube 数据集使用 parquet 格式而不是 JSONL 格式存储元数据。

**修复位置：** `gr00t/data/dataset.py`

**修复方案：**

#### 2.1 修复 `_get_trajectories` 方法
```python
def _get_trajectories(self) -> pd.DataFrame:
    """Get the trajectories for the dataset."""
    episodes_path = self.dataset_path / LE_ROBOT_EPISODES_FILENAME
    
    # Try to load as JSONL first (fallback to parquet if not found)
    if episodes_path.exists():
        return pd.read_json(episodes_path, lines=True)
    else:
        # Try parquet format
        parquet_path = self.dataset_path / "meta/episodes.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        else:
            # Try chunked parquet format
            chunked_path = self.dataset_path / "meta/episodes/chunk-000/file-000.parquet"
            if chunked_path.exists():
                return pd.read_parquet(chunked_path)
        raise FileNotFoundError(f"Could not find episodes data in JSONL or parquet format")
```

#### 2.2 修复 `_get_tasks` 方法
```python
def _get_tasks(self) -> pd.DataFrame:
    """Get the tasks for the dataset."""
    tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
    
    # Try to load as JSONL first (fallback to parquet if not found)
    if tasks_path.exists():
        return pd.read_json(tasks_path, lines=True)
    else:
        # Try parquet format
        parquet_path = self.dataset_path / "meta/tasks.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        else:
            # Try chunked parquet format
            chunked_path = self.dataset_path / "meta/tasks/chunk-000/file-000.parquet"
            if chunked_path.exists():
                return pd.read_parquet(chunked_path)
        raise FileNotFoundError(f"Could not find tasks data in JSONL or parquet format")
```

### 3. KeyError for chunk_index in Video Path

**问题描述：**
```python
KeyError: 'chunk_index'
```

**原因：**
`get_video_path` 方法使用了错误的参数名格式化视频路径。

**修复位置：** `gr00t/data/dataset.py` 第 718-735 行

**修复方案：**
```python
def get_video_path(self, trajectory_id: int, key: str) -> Path:
    original_key = self.lerobot_modality_meta.video[key].original_key
    if original_key is None:
        original_key = key
    
    # Get episode info to extract chunk_index and file_index
    if self._episodes_df is None:
        self._episodes_df = pd.read_parquet(self.dataset_path / "meta/episodes/chunk-000/file-000.parquet")
    
    trajectory_index = self.get_trajectory_index(trajectory_id)
    episode_info = self._episodes_df.iloc[trajectory_index]
    chunk_index = episode_info["data/chunk_index"]
    file_index = episode_info["data/file_index"]
    
    video_filename = self.video_path_pattern.format(
        video_key=original_key, chunk_index=chunk_index, file_index=file_index
    )
    return self.dataset_path / video_filename
```

### 4. Video Decoding Issues

**问题描述：**
```python
ValueError: No valid stream found in input file.
```

**原因：**
pickcube 数据集的视频使用 AV1 编码，但默认的 `torchcodec` 后端不支持 AV1 解码。

**解决方案：**
使用 `torchvision_av` 视频后端替代 `torchcodec`。

**修复方式：** 在训练命令中添加参数
```bash
--video-backend torchvision_av
```

### 5. KeyError for 'task' Column

**问题描述：**
```python
KeyError: 'task'
```

**原因：**
pickcube 数据集的 tasks DataFrame 中任务描述存储在索引中，而不是 'task' 列中。

**修复位置：** `gr00t/data/dataset.py` 第 898-910 行

**修复方案：**
```python
for i in range(len(step_indices)):
    task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())

# Handle different task data formats
if "task" in self.tasks.columns:
    return self.tasks.loc[task_indices]["task"].tolist()
else:
    # If task is in the index (like in pickcube dataset)
    return self.tasks.index[task_indices].tolist()
```

### 6. Dataset Initialization Parameters

**问题描述：**
数据集初始化时参数名称不匹配。

**修复位置：** `scripts/gr00t_finetune.py` 第 210-225 行

**修复方案：**
```python
dataset = LeRobotSingleDataset(
    dataset_path=Path(dataset_path),
    modality_configs=data_config.modality_configs,  # 修正参数名
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,   # 修正参数名
    video_backend=config.video_backend,
    transforms=transforms,
    video_backend_kwargs=None,
    cache_dir=None,
    offline=True,
)
```

## 成功的训练命令

### 基础训练命令
```bash
cd /home/shenlan/GR00T-VLA/Isaac-GR00T
conda activate gr00t

python scripts/gr00t_finetune.py \
  --no-tune_diffusion_model \
  --dataset-path /home/shenlan/.cache/huggingface/lerobot/gr00t/pickcube \
  --num-gpus 1 \
  --batch-size 8 \
  --output-dir ~/checkpoints/pickcube/ \
  --data-config unitree_g1 \
  --max-steps 200 \
  --dataloader-num-workers 2 \
  --video-backend torchvision_av
```

### 推荐参数说明
- `--video-backend torchvision_av`: 必需，支持 AV1 编码视频
- `--batch-size 8`: 平衡内存使用和训练效率
- `--dataloader-num-workers 2`: 减少多进程问题
- `--no-tune_diffusion_model`: 只训练 projector，减少计算需求

## 数据集格式兼容性

### 支持的数据格式
1. **Episodes 数据**: JSONL 或 Parquet 格式
2. **Tasks 数据**: JSONL 或 Parquet 格式  
3. **视频编码**: AV1 (需要 torchvision_av 后端)
4. **统计数据**: 支持标量和多维数组

### 数据集路径结构
```
pickcube/
├── meta/
│   ├── info.json
│   ├── stats.json
│   ├── modality.json
│   ├── episodes.parquet (或 episodes.jsonl)
│   └── tasks.parquet (或 tasks.jsonl)
├── data/
│   └── chunk-{chunk_index:03d}/
│       └── file-{file_index:03d}.parquet
└── videos/
    └── {video_key}/
        └── chunk-{chunk_index:03d}/
            └── file-{file_index:03d}.mp4
```

## 训练结果验证

### 成功指标
- ✅ 模型加载成功 (3B 参数)
- ✅ 数据集加载成功 (7894 样本)
- ✅ 训练损失下降 (从 ~1.0 降至 ~0.1)
- ✅ 检查点保存成功
- ✅ WandB 集成正常

### 检查点位置
```
~/checkpoints/pickcube/
├── checkpoint-50/
├── checkpoint-200/
├── config.json
├── model-*.safetensors
├── trainer_state.json
└── training_args.bin
```

## 故障排除

### 常见问题和解决方案

1. **视频解码错误**
   - 确保使用 `--video-backend torchvision_av`
   - 检查视频文件是否损坏

2. **内存不足**
   - 减少 batch_size (如 4 或 2)
   - 减少 dataloader_num_workers (如 1)

3. **模型加载缓慢**
   - 正常现象，3B 参数模型需要时间
   - 可以使用更小的 max_steps 进行测试

4. **数据格式错误**
   - 检查数据集路径是否正确
   - 确认元数据文件存在

## 环境要求

- Python 3.10+
- PyTorch 2.5.0+cu124
- CUDA 12.4
- GR00T conda 环境
- 足够的 GPU 内存 (>8GB 推荐)

## 7. 评估脚本修复

**问题描述：**
评估脚本 `eval_policy.py` 在使用微调后的模型时遇到数据加载问题。

**原因：**
评估脚本设置了 `transforms=None`，但数据集需要正确的 transforms 来处理数据。

**修复位置：** `scripts/eval_policy.py` 第 130-140 行

**修复方案：**
```python
# Create dataset
dataset = LeRobotSingleDataset(
    dataset_path=args.dataset_path,
    modality_configs=modality,
    video_backend=args.video_backend,
    video_backend_kwargs=None,
    transforms=modality_transform,  # Use actual transforms instead of None
    embodiment_tag=args.embodiment_tag,
)
```

## 评估命令

### 成功的评估命令
```bash
cd /home/shenlan/GR00T-VLA/Isaac-GR00T
conda activate gr00t

python scripts/eval_policy.py --plot \
   --embodiment_tag new_embodiment \
   --model_path /home/shenlan/checkpoints/pickcube/checkpoint-200/ \
   --data_config unitree_g1 \
   --dataset_path /home/shenlan/.cache/huggingface/lerobot/gr00t/pickcube \
   --video_backend torchvision_av \
   --modality_keys left_arm right_arm
```

**重要提示：**
- 必须使用 `--video_backend torchvision_av` 参数
- 模型加载可能需要较长时间（3B 参数模型）
- 建议先使用预训练模型测试：`--model_path nvidia/GR00T-N1.5-3B`

## 总结

这些修复解决了 GR00T 微调脚本与 pickcube 数据集的兼容性问题，使训练和评估都能够顺利进行。关键修复包括：

1. **数据格式兼容性** (JSONL ↔ Parquet)
2. **视频编码支持** (AV1)
3. **统计数据处理**
4. **任务数据格式处理**
5. **参数名称修正**
6. **评估脚本 transforms 修复**

修复后的脚本能够：
- ✅ 成功处理 pickcube 数据集
- ✅ 进行有效的模型微调
- ✅ 正确评估微调后的模型
- ✅ 生成评估结果和可视化图表