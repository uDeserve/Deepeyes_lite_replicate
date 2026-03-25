# DeepEyes 环境配置记录（本机实践）

本文记录本次复现前的环境配置过程：**做了什么、踩了什么坑、当前状态如何**。  
完整复现计划见 Cursor 计划文档：`~/.cursor/plans/deepeyes_轻量复现_8b85c67c.plan.md`。  
**冒烟跑通流程、尝试与踩坑复盘**：[`SMOKE_RUN_PLAYBOOK.md`](./SMOKE_RUN_PLAYBOOK.md)。

---

## 一、已完成的配置工作

### 1) 存储分工落地

| 内容 | 路径 | 文件系统 | 结论 |
|------|------|---------|------|
| conda 环境 | `/data/lsj/deepeyes/conda_env` | ext4 | 已创建，作为 Python/conda 主环境 |
| 模型/数据/checkpoint/HF 缓存 | `/mnt/raid2/lsj/deepeyes/` | fuseblk (NTFS) | 已创建，专门放大文件 |

RAID 目录如下：

```text
/mnt/raid2/lsj/deepeyes/
├── huggingface/hub
├── huggingface/transformers
├── models/
├── data/
├── checkpoints/
├── logs/
├── ray_tmp/
└── pip_cache/
```

并额外创建了：

```text
/data/lsj/deepeyes/conda_env
/data/lsj/deepeyes/pip_cache
/data/lsj/deepeyes/tmp
```

### 2) 环境变量脚本

已创建并验证 `~/deepeyes_raid_env.sh`，核心变量：

```bash
export DEEPEYES_CONDA_ENV=/data/lsj/deepeyes/conda_env
export DEEPEYES_RAID_ROOT=/mnt/raid2/lsj/deepeyes
export HF_HOME="${DEEPEYES_RAID_ROOT}/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
# datasets 的 map/filter 会对缓存文件 os.chmod；NTFS-fuse 上常 PermissionError，故单独放到 ext4
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/data/lsj/deepeyes/hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"
export DEEPEYES_MODEL_DIR="${DEEPEYES_RAID_ROOT}/models"
export DEEPEYES_DATA_DIR="${DEEPEYES_RAID_ROOT}/data"
export SAVE_CHECKPOINT_DIR="${DEEPEYES_RAID_ROOT}/checkpoints"
export RAY_TMPDIR="${DEEPEYES_RAID_ROOT}/ray_tmp"
export PIP_CACHE_DIR="${DEEPEYES_RAID_ROOT}/pip_cache"

# Judge（OpenAI 兼容）
export LLM_AS_A_JUDGE_BASE="https://yunwu.ai/v1"
export LLM_AS_A_JUDGE_API_KEY="<your_key>"
export LLM_AS_A_JUDGE_MODEL="qwen3.5-plus"
```

### 3) conda + 基础包安装

- conda 环境：`python 3.10.20`
- 已安装：
  - `torch==2.6.0+cu124`
  - `torchvision==0.21.0+cu124`
  - `torchaudio==2.6.0+cu124`
  - `verl==0.2.0.dev`（`pip install -e .`）
  - `vllm==0.8.2`
  - `flash-attn==2.7.2.post1`（来自本机历史离线 wheel）
  - `evaluate`、`openai`、`qwen_vl_utils`、`math_verify` 等 DeepEyes 依赖

离线 wheel 来源（已验证可用）：

```text
/home/lsj/zyw/Vision-SR1/flash_whl/flash_attn-2.7.2.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

---

## 二、踩过的坑（重点）

### 坑 1：conda 环境放 RAID

- **现象**：最初计划把 conda 环境也放 RAID。
- **原因**：`/mnt/raid2` 是 `fuseblk (NTFS)`，不适合 conda（symlink/权限语义不完整）。
- **处理**：改为 `/data`（ext4）放 conda，仅把大文件放 RAID。

### 坑 2：沙箱权限导致 `/data` 安装失败

- **现象**：`pip install` 写 `/data/lsj/...` 时报权限问题。
- **原因**：沙箱模式只允许写工作区。
- **处理**：改为无沙箱权限执行安装命令后恢复正常。

### 坑 3：`vllm` 与 `torch` ABI 组合不兼容

- **现象**：`import vllm` 报错：`vllm/_C.abi3.so: undefined symbol ... parseSchemaOrName`。
- **原因**：最初安装 `torch 2.6.0+cu126`，与当前 `vllm==0.8.2` 组合存在 ABI 不匹配。
- **处理**：重装为 `torch 2.6.0+cu124` 后，`import vllm` 正常。

### 坑 4：`flash-attn` 安装失败（跨设备链接）

- **现象**：安装 `flash-attn` 时出现 `Invalid cross-device link`。
- **原因**：`pip cache` 在 RAID，构建临时目录在另一路径，`rename()` 跨设备失败。
- **处理**：将 `TMPDIR` 与 `pip --cache-dir` 都改到 `/data/lsj/deepeyes/...`（同一 ext4）后安装成功。

### 坑 5：`flash-attn` 版本/ABI 兼容

- **现象**：先前安装 `flash-attn 2.8.3` 后，`import flash_attn` 报 `undefined symbol ... c10::Error...`。
- **原因**：当前 torch/vllm 组合下，2.8.3 wheel 与本机 ABI 不匹配。
- **处理**：切换为你之前可用的离线 wheel `2.7.2.post1+cu12torch2.6cxx11abiFALSE`，导入恢复正常。

### 坑 6：`datasets` 缓存在 NTFS（RAID）上 `chmod` 失败

- **现象**：训练启动早期 `RayTaskError(PermissionError)`，栈在 `datasets` 写缓存后的 `os.chmod`。
- **原因**：`HF_HOME` 在 fuseblk/NTFS 上时，部分语义与 ext4 不一致。
- **处理**：使用 **`HF_DATASETS_CACHE`** 指向 ext4 目录（见上文环境变量块）；详见 [`SMOKE_RUN_PLAYBOOK.md`](./SMOKE_RUN_PLAYBOOK.md) §9.1。

---

## 三、当前状态（截至本次记录）

### 可用

- `import torch`：正常，CUDA 可见
- `import verl`：正常
- `import vllm`：正常
- `import flash_attn`：正常（`2.7.2.post1`）
- 数据子集路径、Judge 环境变量、`main_ppo` 二卡冒烟脚本与 nohup 入口已按会话实践配置；**数据管线**（`datasets` + filter）在日志中可跑通（含 `HF_DATASETS_CACHE` 与 `filter_overlong_prompts_workers=0` 等）。

### 待处理 / 未验证

- **二卡 agent + FSDP + vLLM** 在 **多人共用 GPU**（`nvidia-smi` 可见每卡多 `python` 进程）的条件下，**尚未**在会话记录中得到 **`total_epochs=1` 完整跑通** 的日志；失败形态包括 vLLM **KV 预算非正**、**`wake_up`/cumem OOM**、**PyTorch OOM**。详细时间线与日志文件名见 [`SMOKE_RUN_PLAYBOOK.md`](./SMOKE_RUN_PLAYBOOK.md) §9。

---

## 四、可复用命令

### 环境激活

```bash
source ~/deepeyes_raid_env.sh
conda activate /data/lsj/deepeyes/conda_env
```

### 快速自检

```bash
python -c "import torch, flash_attn, vllm, verl; print(torch.__version__, flash_attn.__version__, vllm.__version__, verl.__version__)"
```

### 目录重建

```bash
mkdir -p /mnt/raid2/lsj/deepeyes/{models,data,checkpoints,logs,huggingface/hub,huggingface/transformers,ray_tmp,pip_cache}
mkdir -p /data/lsj/deepeyes/{conda_env,pip_cache,tmp,hf_datasets_cache}
```
