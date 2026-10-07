# MEMLite Stage1 模型部署指南（新机器）

适用对象：`memlite_stage1_high_100task_v1` / `memlite_stage1_low_100task_v1`
两个正式 run 的最终 checkpoint（ModelScope 仓库 `fduTristin/memlite-stage1`）。

⚠️ 仓库里的 `serve_policy.py` / `serve_policy_mem.py` / `serve_policy_memlite_fm.py`
**都不适配** stage1 checkpoint：前两者要求训练 run 目录带 `.hydra/config.yaml`
（stage1 用 torchrun + 纯 YAML 启动，没有 hydra 目录），后者内置"六帧 planner +
memlite_fm 契约"断言，与 stage1 的 `obs_size=1` 配置冲突。
stage1 请使用 **`scripts/serve_memlite_stage1.py`**（本分支已包含），它直接从
`recipe.json` + checkpoint 重建模型与处理器，并复用 `serve_policy_mem.py` 的
msgpack/WebSocket 协议。

## 步骤

### 0. 硬件/驱动

- ≥1 张 80GB 级 GPU（low 推理约 13GB 显存，high 更低，可同卡跑两个）
- NVIDIA 驱动支持 CUDA 12.8

### 1. 拉代码（本分支）

```bash
git clone -b deploy/memlite-stage1 git@github.com:fduTristin/behavior.git
cd behavior   # REPO 根目录，下文称 $REPO
```

### 2. 建 Python 环境

```bash
uv sync --locked               # 按 uv.lock 锁定版本建 .venv（已含 torch cu128 / websockets / msgpack）
# 需 Python 3.10；若没有 uv：pip install uv
```

### 3. 下载模型资产

```bash
.venv/bin/pip install modelscope   # 如 uv.lock 未含
.venv/bin/modelscope login --token <你的ModelScope_Token>   # 私有仓库需要
mkdir -p /your/deploy/root && cd /your/deploy/root   # 下文称 $DEPLOY_ROOT
modelscope download --model fduTristin/memlite-stage1 --local_dir .
```

得到：

```
$DEPLOY_ROOT/
  high/  low/        # 各自含 recipe.json、latest.json、最终 .pt  checkpoint、eval 记录
  manifests/memlite-stage1-v4-action-bounds/stats.json   # low 归一化统计
  models/memlite-b-final-20260910/B-dataset-stats.json   # high 归一化统计
  models/action_tokenizer.pt                             # 动作 VQ codec
  models/qwen3_5_2b_base_processor/                      # Qwen3.5 处理器
```

下载后建议核对最终 checkpoint 的 SHA-256（与 `high/latest.json` /
`low/latest.json` 中记录一致）：

```bash
sha256sum high/step_00048045_save_0027.pt   # 3683f719ebb0...
sha256sum low/step_00098414_save_0021.pt    # d4d760997802...
```

### 4. 启动 server（无需改任何代码）

```bash
cd $REPO
export PYTHONPATH=$REPO/src
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # 防止启动时访问 HF

# 低层 FM policy（端口 10051）
.venv/bin/python scripts/serve_memlite_stage1.py \
    --branch low  --root $DEPLOY_ROOT --port 10051 --device cuda:0

# 高层 planner（端口 10050，可换个终端/另一张卡）
.venv/bin/python scripts/serve_memlite_stage1.py \
    --branch high --root $DEPLOY_ROOT --port 10050 --device cuda:0
```

`--root` 只需要指到下载目录；脚本会自动把 recipe.json 里记录的
训练机绝对路径（`/data/workspace/wsy/behavior2026/...`）重写到 `$DEPLOY_ROOT` 下。
`--ckpt` 可手动指定其他步数的 checkpoint，默认读 `<branch>/latest.json`。

启动日志出现 `Exact restore verified`（逐张量校验，约 1–3 分钟）和
`Policy server listening on ws://0.0.0.0:<port>` 即就绪。

### 4.1 两个进程、各自独立

high 与 low 是**两个相互独立的 server 进程**，无任何内置编排：

| 进程 | `--branch` | 建议端口 | 显存（实测 A800 80G） | 输出 |
|------|-----------|---------|----------------------|------|
| 高层 planner | `high` | 10050 | ≈12.5 GB | 子目标/结果文本（`cot_text`）+ 动作 chunk |
| 低层 FM policy | `low` | 10051 | ≈12.5 GB | 16 步动作 chunk |

两个进程可以放同一张卡（合计约 25GB < 80GB），也可以
`--device cuda:0` / `cuda:1` 分卡。**两者之间的衔接（何时 replan、把 planner
的子目标文本传到低层）由客户端编排**，与官方评测时 bridge 的做法一致：
拿到高层的子目标文本后，拼进低层请求的 `obs["task"]`（或用 `"任务 [PLAN] 子目标"`
格式由 server 自动拆分出 `plan` 字段）。本脚本只保证两端模型各自可用，
不内置这条编排循环。

### 4.2 观测协议要点

客户端以 msgpack 发送观测 dict（详见 `docs/deployment/serve_policy_mem_zh.md`）：

- 必填：`images`（shape_meta 定义的三个相机键，`[C,H,W]` uint8）、
  `state`（见同文档的 state 键列表）、`task`（任务指令字符串）
- 可选：`plan`、`coarse_task`、`frequency`；`task` 中含 `[PLAN]` 时自动拆出 `plan`
- 响应：`{"action": ..., "need_obs": ...}`；predict_cot 模型（high）额外返回 `cot_text`
- `--action_steps 1` 为逐帧实时（RTC）模式，默认 16 为 chunk 复用模式

```bash
# RTC 模式示例
.venv/bin/python scripts/serve_memlite_stage1.py \
    --branch low --root $DEPLOY_ROOT --port 10051 --action_steps 1
```

### 5. 自检客户端

```python
import asyncio, msgpack, websockets
async def t():
    async with websockets.connect("ws://127.0.0.1:10051", max_size=None) as ws:
        print(msgpack.unpackb(await ws.recv()))          # {'action_steps': 16}
        await ws.send(msgpack.packb({"__reset__": True}))
        print(msgpack.unpackb(await ws.recv()))          # {'__reset__': True}
asyncio.run(t())
```

观测/动作的完整 msgpack 协议见 `docs/deployment/serve_policy_mem_zh.md`
（本脚本与其 handler 完全一致：客户端发 obs dict，server 返回
`{"action": ..., "need_obs": ...}`，`__reset__` 清空 episode 缓存）。

## 常见问题

- **缺 `stats.json` / `B-dataset-stats.json` / `action_tokenizer.pt`**：第 3 步
  下载不完整，四个运行时资产缺一不可（high/low 的归一化统计文件不同，不能混用）。
- **正式 BEHAVIOR 评测桥接**（robodojo bridge、官方任务指令列表）不在本仓库内；
  本 server 是裸 policy server，官方评测需另接团队的 bridge 服务（见下节"bridge 说明"）。

## bridge 说明（官方评测才需要）

- bridge 已随本分支 vendor 在 `scripts/behavior_bridge/serve_behavior_policy_mem.py`
  （来源：robodojo `/mnt/sdc1/robodojo/behavior_bridge_staging/`，2026-09-03 验证快照）。
  提供官方 BEHAVIOR v3.9.x 协议：61 维 proprio 切片、三相机 CHW uint8、23 维动作、
  fire-and-forget reset、`/healthz`、官方 bytes-key msgpack 编码。
- 兼容性已验证：配套 CPU 测试（修补过时的测试桩后）在本分支全部通过——
  ```
  cd scripts/behavior_bridge && PYTHONPATH=../..:../../src \
      python test_adapter_mem_cpu.py     # 期望结尾 MEMLITE_BEHAVIOR_ADAPTER_CPU_TESTS=PASS
  ```
- **但 bridge 只解决"wire 协议"层。** 完整官方闭环还需要：
  1. **stage1 planner 运行时**（B-memory K=3 状态机、planner 事件准入、任务切换隔离）：
     团队仓库至今没有发布过 stage1 的官方编排运行时（旧 `serve_policy_memlite_fm.py`
     的六帧/契约断言与 stage1 的 `obs_size=1`、planner-outcome 六字段协议不兼容），
     这部分需要按 `g05.utils.memlite_skill_protocol` 的协议规范新写适配；
  2. **官方 100 任务指令表**：本机已有
     `datasets/2026-challenge-demos/datasets/fduTristin--2026-challenge-demos/snapshots/master/meta/tasks.jsonl`
     （100 行，`task_index`/`task` 字段与 bridge 读取器匹配；任务名与 stage1 发布
     manifest 的 task_names 逐一对应）。部署时把这个文件拷到新机器，
     用 `--tasks_path` 指向即可，无需再向队友索取；
  3. **初始 memory（已解决）**：stage1 的 planner 记忆不是 bridge 快照里那张
     5 任务 `Task=<id>; Completed=none.` 表（那是旧 v9 协议，不要用于 stage1），
     而是 B-memory 规范 JSON。`scripts/memlite_stage1_runtime.py` 按训练侧
     `memlite_stage1_labels.projection` 的公式对全部 100 个任务生成初始记忆，
     `scripts/memlite_stage1_tasks.json` 钉住 100 个 canonical 任务名；
     CPU 契约测试在 `tests/test_stage1_runtime_memory.py`。
```

**planner runtime（进行中）**：以 `scripts/memlite_stage1_runtime.py` 的
memory 原语为状态机核心，按 `g05.utils.memlite_skill_protocol` 与
`g05_policy_memlite_planner_outcome.generate_high_level` 的公开契约拼装
serving 版 planner 运行时；完成后作为完整官方评测闭环的入口。
- 只做策略推理/对接自有客户端 → 用 `serve_memlite_stage1.py`，不需要 bridge。
- 端口习惯沿用：高层 10050，低层 10051。
- 不需要把训练数据集拷到新机器：100 个任务的 prompt 模板已固化在 `recipe.json`。
