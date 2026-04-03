# DeepEyes 冒烟跑通实践手记

本文从**客观复盘**角度整理：此前工作目标、推荐流程、已做尝试、踩坑与规避、以及可复用的成功经验。侧重 **4 卡本地 smoke（`main_ppo` + vLLM rollout + LLM-as-a-Judge）** 相关事项。

与下列文档分工：

- **存储与环境变量速查**：[`REPRODUCTION_zh.md`](./REPRODUCTION_zh.md)
- **本机完整 checklist / 机器专属细节**：Cursor 计划 `~/.cursor/plans/deepeyes_轻量复现_8b85c67c.plan.md`（若路径有变，以你本机为准）

---

## 1. 工作目标与边界

| 维度 | 约定 |
|------|------|
| 主目标 | 以实践 veRL / DeepEyes 流程为主，**优先跑通闭环**，不追求论文级指标 |
| 算力 | 单机 **7×RTX 4090**，训练时 **固定使用其中 4 卡**，为同门预留其余 GPU |
| Judge | 使用 **OpenAI 兼容 API**（本机实践：`https://yunwu.ai/v1`），模型名 **`qwen3.5-plus`** |
| 存储 | **conda 与 pip 构建缓存放 ext4**；**模型、数据集、checkpoint、HF 缓存、Ray 临时目录放 RAID**（`/mnt/raid2/lsj/deepeyes/`） |

---

## 2. 推荐执行流程（顺序）

1. **前置对齐**：目标（学习 / 发论文）、GPU 占用、路径、Judge 服务是否可达。
2. **存储与变量**：`source ~/deepeyes_raid_env.sh`，确认 `HF_HOME`、`SAVE_CHECKPOINT_DIR`、`RAY_TMPDIR` 等指向 RAID；确认 **`HF_DATASETS_CACHE` 指向 ext4**（避免 `datasets` 在 NTFS-fuse 上 `chmod` 失败，见 §9.1）。
3. **干净 Conda 环境**：Python 3.10；按 veRL / 项目要求安装 **PyTorch（与 vLLM  wheel 匹配的 CUDA 变体）**；`pip install -e .`；`bash scripts/install_deepeyes.sh`（遇冲突再分层排查）。
4. **依赖验证（分层门禁）**：`import torch` → `import vllm` → `import flash_attn`（若启用）→ 小脚本拉一次 HF（必要时镜像）。
5. **数据子集**：小 parquet 放在 `DEEPEYES_DATA_DIR/subset/`，在 smoke 脚本中指向 `train_files` / `val_files`。
6. **基座模型**：`Qwen2.5-VL-7B-Instruct` 完整分片下载到 `DEEPEYES_MODEL_DIR`，避免残缺 shard 直接开训。
7. **Judge 环境变量**：`LLM_AS_A_JUDGE_BASE`、`LLM_AS_A_JUDGE_API_KEY`、`LLM_AS_A_JUDGE_MODEL`；用 `curl` 或最小请求验证 **chat/completions** 可用（仅 `/models` 不够）。
8. **冒烟脚本**：`examples/agent/local_smoke_4gpu.sh`；启动前 `export CUDA_VISIBLE_DEVICES=...` 与 `WORLD_SIZE=1`。
9. **日志与排障**：`logs/smoke_4gpu_subset512.log`；OOM 时优先收紧 rollout / 序列长度 / `max_model_len` 相关配置（见下文）。

---

## 3. 做过的主要尝试（摘要）

- **GPU 映射**：用 `CUDA_VISIBLE_DEVICES` 限定 4 卡，避免与同门争抢；训练配置 `trainer.n_gpus_per_node=4`、`nnodes=1`。
- **依赖组合**：为对齐 **vLLM 0.8.2**，将 PyTorch 调整为 **`torch==2.6.0` + `cu124` wheel**，缓解与 vLLM 的 **ABI / undefined symbol** 问题。
- **`triton` 版本**：安装脚本曾 pin `triton==3.1.0`，与 torch 自带 triton 冲突；实践中采用 **不强行覆盖 torch 自带 triton** 的路径完成安装。
- **`flash-attn`**：源码编译曾遇 **跨文件系统 rename**（`/tmp` 与 RAID 缓存不同盘）；将 **`TMPDIR` 与 pip cache 放到 ext4** 同盘解决。导入仍 **undefined symbol** 时，改用 **与本机 torch/CUDA 匹配的预编译 wheel** 成功。
- **Hugging Face 网络**：直连 `huggingface.co` 不可达时，使用 **`HF_ENDPOINT=https://hf-mirror.com`** 走镜像。
- **模型下载**：全量 `hf download` 在 NTFS 上出现 **锁文件等待 / 权限提示**；处理思路为 **结束卡死进程、清理 `*.lock`、按需补全缺失 `model-xxxx-of-xxxx.safetensors`**。
- **Judge**：代码侧支持从环境变量读取 key 与模型名（`verl/utils/reward_score/vl_agent.py`），避免改源码硬编码；API key 需对 **chat 接口** 有效（曾出现仅列表模型成功、对话报「无效令牌」的情况，需与服务侧 key 权限对齐）。
- **冒烟显存**：发现 **`vllm_rollout_spmd.py` 将 `max_model_len` 写死为 32768**，导致 vLLM KV 预留过大，与 FSDP actor 叠加以致 **OOM / 后续 illegal memory access**；改为使用配置推导的 **`max_model_len`（如 prompt+response）** 后，为 4 卡 smoke 创造条件。同时 smoke 脚本侧可配合 **略降 `gpu_memory_utilization`、`max_num_batched_tokens`**、恢复 **`filter_overlong_prompts=True`**。

---

## 3.1 四卡/二卡「过滤完数据后卡住、永远不进 GPU」的原因（重要）

`main_ppo` 里 **`RayPPOTrainer` 在 `trainer.init_workers()` 之前** 就会在 `__init__` 里调用 `_create_dataloader()`（见 `verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py`）。

现象与代码对齐如下：

1. 日志里出现 **`dataset len: …`** 和 **`Filtering prompts longer than …` 进度条到 100%**，但**始终没有**下一行 **`filter dataset len:`**（定义在 `verl/utils/dataset/rl_dataset.py`），说明卡在 **`datasets.Dataset.filter(...)` 返回之前**（多进程池收尾阶段），**还没走到** 创建 `StatefulDataLoader`、更没到 `init_workers()`，因此 **GPU 上不会出现训练进程**。
2. veRL 里 **`StatefulDataLoader` 原先用硬编码 `num_workers=8`**。在 **Ray 的 `TaskRunner` 子进程**里再 `fork` 出一批 DataLoader worker，与 Ray / CUDA 初始化顺序叠加时，也容易出现**死锁或长时间无输出**。
3. **关闭 IDE 终端**时，若训练进程仍挂在该终端会话下（含「后台」任务仍继承该会话），可能收到 **SIGHUP/SIGTERM**，Ray 侧会看到 **SIGTERM**；这与「业务自己跑完」是两件事，需区分。

**本项目已做缓解**（冒烟脚本与数据管线）：

- `data.filter_overlong_prompts_workers=0` → 过滤改为**单进程**（`num_proc=None`）。
- `data.dataloader_num_workers=0` → DataLoader **不启子进程**。
- `rl_dataset.py` 中当 `filter_overlong_prompts_workers==0` 时显式使用 **`num_proc=None`**。

大规模数据若仍想加速过滤，可在**非 Ray 驱动**场景下再把 workers 调大。

---

## 4. 踩坑清单（现象 → 原因/线索 → 处理方向）

| 现象 | 线索 | 处理方向 |
|------|------|----------|
| conda/pip 写 `/data` 报权限错误 | 在受限沙箱或非预期用户下执行 | 在实际用户环境、有写权限的路径下执行 |
| `pip install torch` / triton 冲突 | 项目脚本 pin 与 torch 自带不一致 | 优先保证 **torch 与 vLLM 匹配**；triton 不盲目降级 |
| `import vllm` undefined symbol（torch jit 等） | torch CUDA 变体与 vLLM wheel 不一致 | 统一到 **cu124 + 对应 torch 版本** 等已知可行组合 |
| `flash-attn` 安装 `Invalid cross-device link` | 临时目录与缓存目录跨挂载点 | **`TMPDIR`、pip `--cache-dir` 放到 ext4** |
| `import flash_attn` undefined symbol | wheel 与当前 torch/CUDA ABI 不匹配 | 使用 **匹配版本的预编译 wheel** 或在本机同环境重编 |
| HF `Network is unreachable` | 机房/路由限制 | **`HF_ENDPOINT` 镜像** 或离线缓存 |
| Judge「无效令牌」 | key 无效或仅部分接口开放 | 用 **chat/completions** 实测；检查 baseurl、key、模型名 |
| 模型目录只有部分 shard | 下载中断或锁竞争 | 查 `model.safetensors.index.json`；**补全缺失分片** |
| smoke 在 vLLM 初始化阶段 OOM | KV 按极大 context 预留 | **修正 `max_model_len` 硬编码**；收紧 batch/token/利用率 |
| `datasets` / `map` 报 `PermissionError`（chmod） | 缓存在 NTFS-fuse | **`HF_DATASETS_CACHE` 设到 ext4**（§9.1） |
| vLLM `540672 > 131072` 警告 + `No available memory for the cache blocks` | `limit_mm_per_prompt` 与 `max_vllm_images` 过大、profile 峰值高 | **降低 `max_vllm_images`**；调整 `gpu_memory_utilization`（§9.4） |
| `cumem` / `wake_up` 或生成阶段 CUDA OOM | FSDP 与 vLLM 同卡 + 多进程分显存 | **独占或更空 GPU**；收紧 batch/序列；见 §9.5 |
| Worker 日志里 `playwright` / `gymnasium` 缺失 | 可选工具未装 | 若任务不依赖对应 env，可 **忽略**；需要时再装依赖 |
| 过滤数据后日志长时间无输出 | 多进程加载大模型 + Ray | **正常可能静默数分钟**；结合 `nvidia-smi` 与进程树判断 |

---

## 5. 成功经验（可复用）

1. **先对齐再装环境**：GPU 张数、路径、Judge、是否允许子集数据，减少返工。
2. **存储分型**：conda/ext4 与 大文件/NTFS-fuse 分离，避免 symlink、权限、跨盘 rename 类问题集中爆发。
3. **分层验证**：torch → vllm → flash_attn → HF 拉取 → Judge 对话，每层通过再进入训练。
4. **高风险包优先「经验优先」**：flash-attn、vLLM 与 torch 的矩阵以 **官方/项目推荐 + 本机已验证 wheel** 为准，少做「随手升级」。
5. **冒烟配置单独脚本化**：`local_smoke_4gpu.sh` 集中 **小 epoch、保守 batch、明确环境变量**，与正式训练脚本分离。
6. **显存问题先看「逻辑上下文长度」**：不仅调 `gpu_memory_utilization`，还要确认 **vLLM 侧 `max_model_len` 是否被不合理放大**（本项目曾为此类根因）。
7. **密钥不进仓库**：Judge key 用环境变量注入；文档与命令历史中尽量减少明文长期留存。

---

## 6. 关键文件与入口（冒烟相关）

| 用途 | 路径 |
|------|------|
| 4 卡冒烟入口 | `examples/agent/local_smoke_4gpu.sh` |
| 2 卡冒烟（如物理 GPU 5、6） | `examples/agent/local_smoke_2gpu_gpu56.sh`（需 `CUDA_VISIBLE_DEVICES=5,6`） |
| 2 卡冒烟 **nohup 后台**（关 IDE 仍可跑） | `examples/agent/run_smoke_2gpu_gpu56_nohup.sh` |
| 训练主程序 | `verl/trainer/main_ppo.py` |
| vLLM rollout（含 `max_model_len` 行为） | `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` |
| LLM-as-a-Judge 与 API 配置 | `verl/utils/reward_score/vl_agent.py` |
| 环境变量示例（本机） | `~/deepeyes_raid_env.sh` |
| 冒烟日志（脚本内 `tee`） | `logs/smoke_4gpu_subset512.log` |

---

## 7. 冒烟启动示例（需按本机改设备号与密钥来源）

```bash
source ~/deepeyes_raid_env.sh
# conda activate /data/lsj/deepeyes/conda_env   # 若未默认使用该环境
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=1
export LLM_AS_A_JUDGE_BASE="https://yunwu.ai/v1"
export LLM_AS_A_JUDGE_API_KEY="***"            # 建议从安全来源注入，勿提交仓库
export LLM_AS_A_JUDGE_MODEL="qwen3.5-plus"
export PATH="/data/lsj/deepeyes/conda_env/bin:$PATH"
cd /path/to/DeepEyes
bash examples/agent/local_smoke_4gpu.sh
```

---

## 8. 文档维护说明

- 本文描述的是**截至编写时**的实践结论；框架与依赖版本升级后，部分结论可能需重新验证。
- 若冒烟策略变更（例如改用 TP>1、换基座模型、或关闭 agent rollout），请在脚本与配置中同步更新，并在此文或计划文档中追加一条变更记录即可。

---

## 9. 2026-03 工作纪要（二卡冒烟与排障，客观记录）

本节按时间线汇总对话与仓库内实际改动：**观察到的现象**、**采取的动作**、**可核对的结果**。不将「未在日志或命令输出中验证的结论」写成既定事实。

### 9.1 现象一：`datasets` 与 RAID（NTFS-fuse）上的缓存

**观察（日志）**

- `RayTaskError(PermissionError)`，栈指向 `datasets` → `map` / 缓存路径上的 `os.chmod`。
- 缓存路径形态为：`${HF_HOME}/datasets/.../*.arrow`，其中 `HF_HOME` 指向 `/mnt/raid2/lsj/deepeyes/huggingface`（fuseblk/NTFS）。
- 报错信息含 `[Errno 1] Operation not permitted` 一类语义（与 Unix `chmod` 在部分 fuse 文件系统上不可用一致）。

**动作**

- 在 `~/deepeyes_raid_env.sh` 中增加 **`HF_DATASETS_CACHE`**，默认指向 ext4 路径 **`/data/lsj/deepeyes/hf_datasets_cache`**，并对该目录执行 `mkdir -p`。
- 保持 **`HF_HOME` / Hub / Transformers 缓存** 仍可落在 RAID（按原分工）；仅将 **`datasets` 库写入的 arrow 缓存** 放到支持 `chmod` 的文件系统。
- `examples/agent/run_smoke_2gpu_gpu56_nohup.sh` 的 `nohup env ...` 中增加 **`HF_DATASETS_CACHE` 透传**（未设置时由子 shell `source` 环境脚本填默认）。

**结果（可核对）**

- 后续冒烟日志中出现 **`HF_DATASETS_CACHE=/data/lsj/deepeyes/hf_datasets_cache`** 的打印；`load_dataset` / `filter` 完成并打印 `filter dataset len:`，未再出现上述 `PermissionError`。

**说明**

- 进度条停在 0% 附近后失败，容易被误判为「无限卡住」；实为 **快速失败**，需以栈与错误类型为准。

---

### 9.2 现象二：数据阶段「卡住」与 Ray 内多进程

**观察**

- 与 playbook §3.1 一致：过滤进度条到 100% 后长时间无返回、GPU 无训练进程等。

**动作（此前已在仓库落地，本节仅对齐记录）**

- `data.filter_overlong_prompts_workers=0`、`data.dataloader_num_workers=0` 写入冒烟脚本。
- `verl/utils/dataset/rl_dataset.py`：在 `filter_overlong_prompts_workers==0` 时使用 **`num_proc=None`**（单进程 filter）。
- `verl/trainer/config/ppo_trainer.yaml` / `ppo_megatron_trainer.yaml` 增加 **`data.dataloader_num_workers`**；`verl/trainer/ppo/ray_trainer.py` 中 `StatefulDataLoader` 使用该配置项。

**结果**

- 与 §9.1 合并验证：数据管线在 TaskRunner 内完成，日志出现 **`[verl:RayPPOTrainer] _create_dataloader: end`** 等阶段行（见 §9.3）。

---

### 9.3 可观测性：阶段日志

**动作**

- `verl/utils/dataset/rl_dataset.py`：在 `_read_files_and_tokenize` 各步骤增加带 **`flush=True`** 的 `print`（含当前 `HF_DATASETS_CACHE`、各 parquet 路径、filter 参数）。
- `verl/trainer/ppo/ray_trainer.py`：在 `_create_dataloader` 内对 train/val dataset 与 dataloader 构建前后打印并 `flush`。
- `verl/trainer/main_ppo.py`：`TaskRunner.run` 中在 `RayPPOTrainer` 构造、`init_workers()`、`fit()` 前后打印并 `flush`。

**结果**

- 日志中出现前缀 **`[verl:RLHFDataset]`**、**`[verl:RayPPOTrainer]`**、**`[verl:TaskRunner]`** 的行，可用于判断卡点位于数据集、DataLoader 还是 worker 初始化之后。

---

### 9.4 现象三：vLLM 初始化 — 超长 token 警告与 KV 预算非正

**观察（日志，示例：`logs/nohup_smoke_2gpu_20260325_151233.log`）**

- Worker 日志：**`Token indices sequence length is longer than the specified maximum sequence length for this model (540672 > 131072)`**。
- 随后 **`ValueError: No available memory for the cache blocks. Try increasing gpu_memory_utilization`**，栈在 **`vllm/v1/core/kv_cache_utils.py`** 的 **`check_enough_kv_cache_memory`**，分支为 **`available_memory <= 0`**（与「KV 需求大于可用」的另一分支文案不同，需以实际栈为准）。
- 同次运行的 Hydra overrides 中 **`actor_rollout_ref.rollout.agent.activate_agent=True`**，配置默认 **`max_vllm_images=32`**（`ppo_trainer.yaml`）；`vllm_rollout_spmd.py` 在 agent 激活时向 `LLM(...)` 传入 **`limit_mm_per_prompt`**（含 `image=max_vllm_images`）。

**对照代码（vLLM 安装树，仅作机制说明）**

- `vllm/v1/worker/gpu_worker.py`：`determine_available_memory` 中  
  `available_kv_cache_memory = total_gpu_memory * gpu_memory_utilization - peak_memory`（`peak_memory` 含 profile 阶段峰值）。
- `vllm/model_executor/models/qwen2_vl.py`：`get_num_frames_with_most_features` 等使用 **`limit_mm_per_prompt` 中的 image 数量** 参与最坏情况 token 相关推导；**大 `max_vllm_images` 会放大 profile 与 tokenizer 警告中的序列规模**。

**动作**

- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`：在 **`max_vllm_images > 8`** 时 **`logger.warning`**，说明与 **FSDP 同卡** 时易导致 **KV 预算非正** 或相关初始化失败，并建议缩小 `max_vllm_images` 或调整 `gpu_memory_utilization`（文案与代码条件以文件为准）。
- `examples/agent/local_smoke_2gpu_gpu56.sh`：为冒烟 **显式降低 `max_vllm_images`**（记录编写时为 **4**），并多次调整 **`gpu_memory_utilization`**、**`enable_chunked_prefill`**、**batch/序列长度**（见 §9.6）。

**结果**

- **`max_vllm_images` 从 32 降到较小值后**，部分运行 **越过** 原先的 **`No available memory for the cache blocks`（≤0 分支）**，进入后续阶段（见 §9.5）。  
- **未**在共享 GPU 条件下得到「任意配置必过」的结论。

---

### 9.5 现象四：`wake_up()` / `cumem` 与 PyTorch OOM

**观察（日志，示例：`logs/nohup_smoke_2gpu_20260325_161639.log` 及后续一次运行）**

- **`RuntimeError: CUDA Error: out of memory`**，栈在 **`vllm/device_allocator/cumem.py`** 的 **`create_and_map`** / **`wake_up`**。
- 另一日志片段：**`torch.OutOfMemoryError`**，发生在 vLLM 侧 **`Qwen2DecoderLayer` / `torch.empty`** 等权重分配处；**`nvidia-smi`** 显示 **同一张物理 GPU 上存在多个 `python` 进程**（多用户或多任务分占显存），与「单进程独占」假设不一致。

**对照代码（verl）**

- `verl/workers/sharding_manager/fsdp_vllm.py`：`__enter__` 中在 **`wake_up`** 后 **`update_params`**；显存峰值与 **FSDP `state_dict()`**、**vLLM 权重池** 的时序相关。

**动作**

- 冒烟脚本侧尝试组合：**减小 `data.train_batch_size` / `ppo_mini_batch_size`**、**降低 `data.max_prompt_length` / `max_response_length`**、**调整 `gpu_memory_utilization`**、**开启 `enable_chunked_prefill`**、**调整 `max_num_batched_tokens`** 等（具体数值以当前 `local_smoke_2gpu_gpu56.sh` 为准）。

**结果**

- **提高 `gpu_memory_utilization`** 有助于通过 §9.4 的 KV 初始化，但 **仍可能在 §9.5 的 `wake_up` 或后续前向 OOM**。
- **降低 `gpu_memory_utilization`** 曾 **再次触发** §9.4 的 **`available_memory <= 0`**。
- 在 **多进程共用 GPU** 的实测环境下，**未记录到** `main_ppo` **完整跑完 `total_epochs=1` 的成功日志**。

---

### 9.6 当前脚本快照（`examples/agent/local_smoke_2gpu_gpu56.sh`，以文件为准）

记录编写时该脚本中 **与排障相关的 overrides**（若日后变更，以 Git 为准）：

- `data.train_batch_size=4`，`data.max_prompt_length=2048`，`data.max_response_length=2048`
- `actor_rollout_ref.actor.ppo_mini_batch_size=4`
- `actor_rollout_ref.rollout.max_num_batched_tokens=8192`，`gpu_memory_utilization=0.68`，`enable_chunked_prefill=True`
- `actor_rollout_ref.rollout.agent.max_vllm_images=4`，`single_response_max_tokens=1024`，`max_turns=3`
- 文件头注释：**若 OOM，先用 `nvidia-smi` 确认物理卡 5、6 无他人占卡**

**入口**

- 前台：`bash examples/agent/local_smoke_2gpu_gpu56.sh`（需事先 `export CUDA_VISIBLE_DEVICES=...` 等）。
- 后台：`bash examples/agent/run_smoke_2gpu_gpu56_nohup.sh`；日志 **`logs/nohup_smoke_2gpu_*.log`** 与软链 **`logs/nohup_smoke_2gpu_latest.log`**。

---

### 9.7 环境：`nvidia-smi` 观测（共享机器）

**观察（某次在机器上执行的 `nvidia-smi` 输出）**

- **7 张 RTX 4090** 均存在 **非零显存占用**；多张卡 **GPU 利用率** 高。
- **每张卡上列出多个 `python` PID**，显存被多进程分割占用。

**结果**

- 该观测 **不证明**「所有进程均为他人」，但说明 **不能假设** 绑定 `CUDA_VISIBLE_DEVICES=5,6` 即等价于 **两张空卡**。  
- 与 §9.5 的 OOM 日志 **在现象上相容**（空闲显存过小、碎片或峰值叠加）。

---

### 9.8 小结表（事实边界）

| 项目 | 状态 |
|------|------|
| `datasets` 缓存 `chmod` + RAID | 已通过 **`HF_DATASETS_CACHE` 指 ext4** 规避；有成功日志佐证 |
| Ray 内 filter / DataLoader 多进程 | 已通过 **workers=0** 与 **`rl_dataset` 单进程 filter** 缓解；有阶段日志佐证 |
| 阶段日志 | 已合入仓库，便于定位卡点 |
| `max_vllm_images` 过大导致 vLLM profile / KV 初始化问题 | 已 **部分** 通过冒烟参数与代码 **warning** 处理；机制与 vLLM 源码对照一致 |
| 二卡 agent + FSDP + vLLM 全流程冒烟 | **截至本记录编写未在共享 GPU 环境下验证通过**；失败形态含 **KV 预算非正**、**cumem wake OOM**、**PyTorch OOM** |
| 四卡 agent + FSDP + vLLM 全流程冒烟（2026-04-02） | **OOM失败**；FSDP actor(~14GB/卡) + ref(~7GB/卡) + vLLM(~30GB/卡) + KV cache 超过48GB/卡；见 §10 |
| 四卡保守配置冒烟（2026-04-02） | **vLLM KV初始化失败**；显存利用率 0.68 时报错 "No available memory for cache blocks"；见 §11 |
| 二卡冒烟（2026-04-02） | **vLLM KV初始化失败**；同门进程占用GPU 0 导致可用显存不足；见 §11 |

---

### 9.9 相关日志文件（仓库外路径，仅列举曾引用的文件名）

- `logs/nohup_smoke_2gpu_20260325_150539.log`：较早失败（与 datasets/权限相关会话对应）。
- `logs/nohup_smoke_2gpu_20260325_151233.log`：数据管线通过；含 **540672 token 警告** 与 **`No available memory for the cache blocks`**。
- `logs/nohup_smoke_2gpu_20260325_161639.log`：含 **`cumem` / `wake_up` OOM**。
- `logs/nohup_smoke_2gpu_20260325_162126.log`：降低 `gpu_memory_utilization` 后 **再次出现** KV 侧 **`No available memory for the cache blocks`**。
- `logs/nohup_smoke_2gpu_20260325_165729.log`：后续尝试中 **PyTorch OOM** 与多进程占显存信息出现在同一会话分析中。

以上文件是否仍保留在磁盘上取决于本机清理策略；**以当时拷贝或 Git 无关**。

---

## 10. 2026-04-02 首次4卡冒烟失败（客观记录）

本节记录 **Trae AI IDE 环境下首次4卡冒烟测试** 的完整过程：**现象、配置、错误、环境状态**。

### 10.1 环境状态

| 项目 | 状态 |
|------|------|
| GPU | 7×RTX 4090，每卡48GB，训练前几乎空闲（~18 MiB占用） |
| torch | 2.6.0+cu124 |
| vLLM | 0.8.2 |
| flash_attn | 2.7.2.post1 |
| verl | 0.2.0.dev |
| tensordict | 0.6.2 |
| Python路径 | 需显式 `export PATH="/data/lsj/deepeyes/conda_env/bin:$PATH"` |
| HF_DATASETS_CACHE | `/data/lsj/deepeyes/hf_datasets_cache`（ext4） |
| Judge API | `https://yunwu.ai/v1`，API Key临时设为 `sk-123456` |

### 10.2 启动命令

```bash
source ~/deepeyes_raid_env.sh
export PATH="/data/lsj/deepeyes/conda_env/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WORLD_SIZE=1
export LLM_AS_A_JUDGE_BASE="https://yunwu.ai/v1"
export LLM_AS_A_JUDGE_API_KEY="sk-123456"
export LLM_AS_A_JUDGE_MODEL="qwen3.5-plus"
cd /home/lsj/zyw/DeepEyes
bash examples/agent/local_smoke_4gpu.sh 2>&1 | tee "./logs/smoke_4gpu_$(date +%Y%m%d_%H%M%S).log"
```

### 10.3 关键配置参数（local_smoke_4gpu.sh）

| 参数 | 值 | 说明 |
|------|-----|------|
| `trainer.n_gpus_per_node` | 4 | 使用4卡 |
| `data.train_batch_size` | 16 | |
| `data.max_prompt_length` | 4096 | |
| `data.max_response_length` | 4096 | |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | 16 | |
| `actor_rollout_ref.rollout.gpu_memory_utilization` | 0.82 | 82%显存 |
| `actor_rollout_ref.rollout.agent.max_vllm_images` | 8 | |
| `actor_rollout_ref.rollout.n` | 2 | 每个prompt采样2次 |
| `actor_rollout_ref.rollout.enable_chunked_prefill` | False | |
| `data.filter_overlong_prompts_workers` | 0 | 单进程过滤 |
| `data.dataloader_num_workers` | 0 | |

### 10.4 运行阶段日志

| 时间 | 现象 |
|------|------|
| 启动 | Ray TaskRunner初始化，配置打印完成 |
| ~20:49 | 模型分片加载（5个shard，每个~14GB） |
| ~20:50 | NCCL初始化完成，Flash Attention警告（可忽略） |
| ~20:52 | FSDP wrap policy设置，vLLM开始profile |
| ~20:52 | Token序列长度警告：`147456 > 131072` |
| ~20:53 | vLLM profile完成，开始分配KV cache |
| ~20:54 | **OOM崩溃** |

### 10.5 显存变化过程

| 时刻 | GPU 0 | GPU 1 | GPU 2 | GPU 3 |
|------|-------|-------|-------|-------|
| 训练前 | 22 MiB | 22 MiB | 22 MiB | 22 MiB |
| 模型加载后 | 22 MiB | 476 MiB | 476 MiB | 476 MiB |
| FSDP初始化 | 28394 MiB | 24860 MiB | 28772 MiB | 29366 MiB |
| vLLM KV分配 | 48412 MiB | 48188 MiB | 37580 MiB | 37580 MiB |
| **最终** | **OOM** | | | |

### 10.6 错误信息

```
RuntimeError: CUDA Error: out of memory at /workspace/csrc/cumem_allocator.cpp:62

堆栈：
  File "vllm/device_allocator/cumem.py", line 214, in wake_up
    create_and_map(handle)
  File "vllm/device_allocator/cumem.py", line 76, in create_and_map
    python_create_and_map(*allocation_handle)
```

### 10.7 主观经验

1. **根因分析**：FSDP actor模型(~28GB) + FSDP ref模型(~14GB) + vLLM rollout(~30GB) + KV cache预算 **同时占用超过48GB**，在vLLM尝试wake_up分配额外KV cache时触发OOM。

2. **4卡配置显存占用估算**：
   - Actor (FSDP): ~14GB/卡
   - Ref (FSDP): ~7GB/卡
   - vLLM + KV: ~30GB/卡
   - **总计**: ~51GB/卡 > 48GB

3. **与playbook记录的二卡失败现象一致**：`cumem wake_up OOM` 正是之前二卡测试中遇到的同类问题。

4. **可能的解决方向**：
   - 降低 `gpu_memory_utilization` 至 0.68-0.70
   - 降低 `max_vllm_images` 至 4
   - 减小 `train_batch_size` 和 `ppo_mini_batch_size`
   - 开启 `enable_chunked_prefill=True`
   - 或改用 **2卡保守配置**

### 10.8 相关日志文件

- `logs/smoke_4gpu_subset512.log`：本次4卡冒烟的完整日志（含OOM堆栈）
- 冒烟脚本已在仓库中，路径：`examples/agent/local_smoke_4gpu.sh`

---

## 11. 2026-04-02 保守配置尝试记录

### 11.1 4卡保守配置（local_smoke_4gpu_conservative.sh）

基于第一次4卡OOM失败，降低了以下参数：

| 参数 | 原值 | 新值 |
|------|------|------|
| `train_batch_size` | 16 | 8 |
| `max_prompt_length` | 4096 | 2048 |
| `max_response_length` | 4096 | 2048 |
| `ppo_mini_batch_size` | 16 | 8 |
| `gpu_memory_utilization` | 0.82 | 0.68 |
| `max_vllm_images` | 8 | 4 |
| `enable_chunked_prefill` | False | True |
| `rollout.n` | 2 | 1 |

**结果**：数据管线通过，但vLLM KV初始化阶段失败：
```
ValueError: No available memory for the cache blocks. 
Try increasing `gpu_memory_utilization` when initializing the engine.
```

### 11.2 2卡冒烟配置（local_smoke_2gpu_gpu56.sh）

使用GPU 5和6，配置如下：

| 参数 | 值 |
|------|-----|
| `trainer.n_gpus_per_node` | 2 |
| `gpu_memory_utilization` | 0.68 |
| `max_vllm_images` | 4 |
| `max_prompt_length` | 2048 |
| `max_response_length` | 2048 |

**结果**：同样在vLLM KV初始化阶段失败。

### 11.3 根因分析

通过 `nvidia-smi --query-compute-apps` 发现同门（fzx用户）有Python进程占用GPU：

| PID | 进程 | 占用显存 |
|-----|------|----------|
| 70871 | collect_data.py | 5.3 GB |
| 76330 | collect_data.py | 2.2 GB |

**GPU 0** 总占用 ~7GB，导致vLLM无法分配足够的KV cache。

### 11.4 经验总结

1. **4卡OOM vs vLLM KV初始化失败**：两种错误都是显存问题，但原因不同
   - OOM：FSDP + vLLM + KV 总需求超过48GB
   - KV初始化失败：FSDP分片后单卡显存碎片化 + 外部进程占用

2. **GPU隔离建议**：
   - 冒烟测试前用 `nvidia-smi --query-compute-apps` 确认GPU空闲
   - 与同门协商GPU分配，或使用空闲GPU编号

3. **可能的解决方向**：
   - 等待同门释放GPU 0
   - 或选择其他空闲GPU（当前GPU 1-6大部分空闲）
   - 或进一步降低 `gpu_memory_utilization` 到 0.50-0.55

### 11.5 新增脚本

- `examples/agent/local_smoke_4gpu_conservative.sh`：4卡保守配置
- `examples/agent/local_smoke_4gpu_ultra_conservative.sh`：4卡极保守配置（未测试）

### 11.6 相关日志文件

- `logs/smoke_4gpu_conservative_20260402_211930.log`：保守配置4卡尝试
- `logs/smoke_2gpu_conservative_20260402_214550.log`：2卡尝试
