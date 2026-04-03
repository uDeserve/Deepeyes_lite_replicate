#!/usr/bin/env bash
set -euxo pipefail

# 4卡冒烟保守配置（2026-04-02调整）
# 针对48GB显存优化，避免OOM
#
# Usage:
# 1) source ~/deepeyes_raid_env.sh
# 2) export PATH="/data/lsj/deepeyes/conda_env/bin:$PATH"
# 3) export CUDA_VISIBLE_DEVICES=0,1,2,3
# 4) export WORLD_SIZE=1
# 5) export LLM_AS_A_JUDGE_BASE="https://yunwu.ai/v1"
# 6) export LLM_AS_A_JUDGE_API_KEY="your-api-key"
# 7) export LLM_AS_A_JUDGE_MODEL="qwen3.5-plus"
# 8) bash examples/agent/local_smoke_4gpu_conservative.sh

PROJECT_NAME="agent_vlagent_local"
EXPERIMENT_NAME="smoke_4gpu_conservative"

: "${DEEPEYES_DATA_DIR:=/mnt/raid2/lsj/deepeyes/data}"
: "${SAVE_CHECKPOINT_DIR:=/mnt/raid2/lsj/deepeyes/checkpoints}"
: "${DEEPEYES_MODEL_DIR:=/mnt/raid2/lsj/deepeyes/models}"
: "${WORLD_SIZE:=1}"
: "${LLM_AS_A_JUDGE_MODEL:=qwen3.5-plus}"

mkdir -p ./logs
mkdir -p "${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}"

SUBSET_TRAIN="${DEEPEYES_DATA_DIR}/subset/data_0.1.2_visual_toolbox_v2.subset_512.parquet"
SUBSET_VAL="${DEEPEYES_DATA_DIR}/subset/data_0.1.2_visual_toolbox_v2.subset_512.parquet"

REF_MODEL_PATH="${DEEPEYES_MODEL_DIR}/Qwen2.5-VL-7B-Instruct"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    data.train_files="[${SUBSET_TRAIN}]" \
    data.val_files="[${SUBSET_VAL}]" \
    data.train_batch_size=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=0 \
    data.dataloader_num_workers=0 \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.checkpoint.contents="['model','hf_model','optimizer','extra']" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.68 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=1024 \
    actor_rollout_ref.rollout.agent.max_turns=3 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.max_vllm_images=4 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes="${WORLD_SIZE}" \
    trainer.save_freq=1000 \
    trainer.test_freq=1000 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    +trainer.tensorboard_dir="${SAVE_CHECKPOINT_DIR}/logs/tensorboard" \
    +trainer.rl_logging_board_dir="${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board" \
    trainer.total_epochs=1 2>&1 | tee "./logs/${EXPERIMENT_NAME}.log"
