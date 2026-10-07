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
- **要跑官方 BEHAVIOR 评测**：不要用本节的裸 server，直接用
  `serve_memlite_stage1_behavior.py`（见下节"官方评测服务"）。

## 官方评测服务（完整闭环，单卡约 25GB）

入口 `scripts/serve_memlite_stage1_behavior.py`，组合三层：

1. **wire 协议**：`scripts/behavior_bridge/serve_behavior_policy_mem.py`
   （vendor 自 robodojo `/mnt/sdc1/robodojo/behavior_bridge_staging/`
   2026-09-03 验证快照）：61 维 proprio 切片、三相机 CHW uint8、23 维动作、
   fire-and-forget reset、`/healthz`、官方 bytes-key msgpack 编码。
   配套官方 100 任务表 vendor 在 `scripts/behavior_bridge/tasks.jsonl`
   （sha256 `8d822231…b8427c`，与 2026-challenge-demos meta 同源，
   部署时无需再向队友拷贝）。
2. **stage1 planner runtime**：`scripts/memlite_stage1_runtime.py` ——
   B-memory K=3 状态机、100 任务初始 memory（B-memory 规范 JSON，与训练侧
   `memlite_stage1_labels.projection` 逐字节一致，经协议校验器全量验证）、
   model projection 构造、事件准入双重复核；任务名资产
   `scripts/memlite_stage1_tasks.json`。
   ⚠️ bridge 快照里那张 5 任务 `Task=<id>; Completed=none.` 表是旧 v9 协议，
   **不要用于 stage1**。
3. **模型加载**：复用 `serve_memlite_stage1.py` 的 recipe 重建（已对全部
   绝对路径做深度重映射，ModelScope 下载后的目录可直接作 `--root`）。

```bash
# 官方 evaluator 直连此端口（默认 10100；不要再用高低分口 10050/10051）
python scripts/serve_memlite_stage1_behavior.py \
    --root /path/to/memlite-stage1 --port 10100 --device cuda:0
# 健康检查：curl http://<开发机>:10100/healthz   → OK
```

**已验证（2026-10-07，a800-2，GPU0 18.7GB）**：模拟官方客户端 18 步在线 +
reset 重启全过；planner 连续 3 个事件全部 `<HL_END>` 闭合、`memory_update`
逐字节满足 K=3 递推；首个 bundle `NAVIGATE→radio_89` 与数据集首个 segment
一致；低层 FM 每 chunk 32 步取前 16。CPU 侧契约测试
`tests/test_stage1_runtime_memory.py` 11/11、bridge adapter 测试全过
（`MEMLITE_BEHAVIOR_ADAPTER_CPU_TESTS=PASS`）。

**遗留提醒**：官方 reset 是 fire-and-forget；评测器若在同一连接内于 reset
后立即发首帧 obs，可能撞上断连（模拟脚本实测如此）——正式评测前请用官方
evaluator 再复核此边界；如需支持再接入容忍。正式评测数值只有在真仿真器里
跑过才可下结论，本仓库内证据不构成成功率结论。

只做策略推理 / 对接自有客户端仍可用裸 server（`serve_memlite_stage1.py`，
原高低分口 10050/10051 的习惯仅适用于它）；100 个任务的 prompt 模板已
固化在 `recipe.json`，部署无需拷贝训练数据集。
