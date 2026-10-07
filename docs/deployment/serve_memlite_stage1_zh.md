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

- bridge = `/mnt/sdc1/robodojo/behavior_bridge_staging/serve_behavior_policy_mem.py`，
  官方 BEHAVIOR 评测桥接：23 维 wire 协议、episode reset、初始 memory 注入、
  官方任务指令（`/mnt/sdc1/xhz/BEHAVIOR2026/2026-challenge-demos/meta/tasks.jsonl`）、
  `/healthz` 检查。`scripts/serve_policy_memlite_fm.py` 通过 `load_bridge()` 动态加载它。
- **该文件从未进入任何 git 分支**，只在 robodojo 集群（队友侧存储）上存在；
  本部署分支的 `serve_memlite_stage1.py` 不依赖 bridge。
- 只做策略推理/对接自有客户端 → 不需要 bridge。
- 要跑官方评测 → 需从队友处拷贝 bridge 目录和 `tasks.jsonl` 到新机器，
  用 `--bridge-dir` / `--tasks_path` 指定路径后使用 `serve_policy_memlite_fm.py` 路线；
  注意该路线还要求 checkpoint 附带六帧 planner 契约，stage1 checkpoint 需先改造适配。
- 端口习惯沿用：高层 10050，低层 10051。
- 不需要把训练数据集拷到新机器：100 个任务的 prompt 模板已固化在 `recipe.json`。
