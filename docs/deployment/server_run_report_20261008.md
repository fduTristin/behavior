# MEMLite Stage 1 闭环 Server 运行状态与剩余条件

- 日期：2026-10-08（北京时间）
- 节点：`/run/ti/BEHAVIOR2026`
- 适配分支：`fix/node-stage1-e2e-20261008`
- 代码提交：`78dcc6145b241d471cd87eb2c5a090ef506794fa`

## 当前结论

完整 high+low server 已经从新分支在 `behavior-server` tmux session 启动，当前运行于 GPU 0 / 端口 `10110`，PID 为 `3445679`，`/healthz` 返回 `OK`。两个 checkpoint 均精确恢复，常驻显存约 18.9 GiB；官方 WebSocket 握手也已通过。

当前不能宣称“正向闭环已经跑通”：全零 synthetic observation 和 radio 纹理替代图两次请求都在 high planner 自由生成阶段达到 1024-token 上限，未生成 `<HL_END>`，严格服务按设计关闭连接，因而没有进入同一 WebSocket 请求的 low chunk。低层已另用同一 low checkpoint 在 GPU 1 直接实测通过：产生 32-step horizon，并成功映射为有限的 `(23,)` official action。剩余缺口已收敛为“用真实 BEHAVIOR evaluator 首帧复验 high AR；若仍不闭合，则处理 high checkpoint 的自由生成/闭合能力”，不再是环境、CUDA、checkpoint restore、tmux、协议或 low FM 问题。

Policy server 本身只需要 CUDA compute 能力，不需要 Vulkan 或 GPU graphics。旧报告中的 graphics capability 问题只阻塞 OmniGibson 仿真，不阻塞 high planner、low FM 和 WebSocket server 的独立验证。

## 当前代码状态

新 worktree 位于：

```text
/run/ti/BEHAVIOR2026/behavior-node-stage1
```

它基于本地可用的 `deploy/memlite-stage1@018cce3` 创建。原 `/run/ti/BEHAVIOR2026/behavior` 未切分支、未 pull、未热改。2026-10-08 12:24 权限生效后，旧的 10050、10051 和 10100 服务均按精确 PID 发送 `TERM`，两秒内正常退出；没有使用宽泛 `pkill` 或 `KILL`。

已完成的本节点适配包括：

- 从本地 Git 对象精确恢复 `src/g05/utils/memlite_planner_format.py`。恢复后的 blob 为 `983692191891662962cda0fcd3ff0541b6977a62`，与残留原始对象一致。
- 支持 recipe 中以标量保存的 state/action `raw_shape`，避免对整数取 `[-1]`。
- 保持官方 bridge 输入的 CHW 图像布局，生成可写的 `[T,C,H,W]` tensor，不再错误转换成 HWC。
- 将 processor 的 canonical 原始字段和 `planner_*` 渲染别名转换为模型要求的 target-free prefix，并把 template 精确截断到 `<EOC>`。
- 在 sm_120 Blackwell 上跳过当前不兼容的 FA4/FA2 wheel，使用 SDPA fallback。
- planner token budget 默认继承 checkpoint recipe 的审核值，不再沿用旧的 160-token 默认值；当前 checkpoint 配置为 1024。
- 新增 `scripts/smoke_official_client.py`，可在不启动 OmniGibson 的情况下验证握手、reset、planner AR、low FM 和 23 维动作响应。

当前分支尚未 push。权限恢复后已于 2026-10-08 重新执行 `git fetch origin --prune` 并成功；`origin/deploy/memlite-stage1` 仍为 `018cce308b27`，与本分支基线一致，远端部署分支没有需要补迁的新提交。

## 已完成验证

| 验证项 | 结果 | 结论边界 |
| --- | --- | --- |
| Stage 1 和 MEMLite 相关 CPU 回归 | 52 项通过 | 覆盖 memory、标签、HL end、processor、runtime 和 conditioning 合同 |
| 官方 bridge 非 socket 回归 | 7 项通过 | 覆盖官方观测、动作映射和 msgpack codec；socket 项受当前沙箱限制未执行 |
| 缺失 planner format 模块 | 导入和单测通过 | receipt、batch、`<HL_END>` token 一致性均有回归 |
| 真实 high recipe 和 processor | 通过 | 三相机输出均为 `[1,3,256,256]` |
| Target-free planner prefix | 通过 | 真实 processor 输出经转换后通过模型自身 `_validate_target_free_high_prefix` |
| 完整 server、healthz、official handshake | 通过 | `behavior-server` / GPU 0 / 10110；握手为 G0.5、R1Pro、23D、16 steps |
| 新分支 high 和 low checkpoint GPU restore | 通过 | high 950 tensors、low 1138 tensors 精确恢复 |
| Planner AR 到 `<HL_END>` | 未通过 | 两个非真实 evaluator 输入均重复 active-skills/memory，1024 token 内不闭合 |
| Low FM 和 23 维动作映射 | 独立通过 | GPU 1 直接测试：horizon 32、shape `(23,)`、全部有限、L2 `1.420634` |
| 同一 official 请求的 planner→low→action | 未通过 | high 严格拒绝后连接关闭，未进入 low；不能用伪 planner 或伪动作冒充闭环 |
| OmniGibson rollout | 未执行 | 另受 graphics capability 阻塞 |

运行证据保存在：

- server：`/run/ti/BEHAVIOR2026/logs/serve_e2e_10110_20261008.log`
- 全零 official client：`/run/ti/BEHAVIOR2026/logs/smoke_official_10110_20261008.log`
- radio 资产 official client：`/run/ti/BEHAVIOR2026/logs/smoke_official_radio_asset_10110_20261008.log`
- low FM 独立 GPU 测试：`/run/ti/BEHAVIOR2026/logs/low_fm_direct_gpu1_20261008.log`

## 2026-10-08 宿主操作记录

12:20 二次核对确认以下三个主进程均属于本项目旧服务：

| PID | 服务 | 设备 | 端口 |
| ---: | --- | --- | ---: |
| `3361174` | `serve_memlite_stage1.py --branch low` | `cuda:0` | `10051` |
| `3361341` | `serve_memlite_stage1.py --branch high` | `cuda:1` | `10050` |
| `3395333` | `serve_memlite_stage1_behavior.py` | `cuda:4` | `10100` |

12:20 的受限 shell 对每个 PID 执行 `kill -TERM` 都返回 `No such process`，但随后从宿主 `/proc` 仍能看到相同 PID、启动时间和命令行。这是当时的 PID namespace 隔离，不是进程已经退出。

同一 shell 的运行能力预检结果为：

- `tmux list-sessions` 连接 `/tmp/tmux-0/default` 返回 `Operation not permitted`。
- 在 workspace 下尝试创建独立 tmux socket同样返回 `Operation not permitted`。
- 共享运行环境是 PyTorch `2.7.1+cu128`，但 `torch.cuda.is_available()` 为 `False`、device count 为 `0`，并且 `/dev/nvidia*` 不存在。
- 创建 IPv4 socket即返回 `PermissionError: Operation not permitted`。

12:24 权限变更生效后重新核对了 PID、命令行和 `nvidia-smi` compute 列表；三者正是当时全部 GPU compute 进程。以下操作已经执行完成：

```bash
for pid in 3361174 3361341 3395333; do
  ps -p "$pid" -o pid=,ppid=,lstart=,args=
done

kill -TERM 3361174 3361341 3395333

# 等待服务清理 TorchInductor 子进程后复核；只有仍存活且命令行未变化时才进一步处理。
sleep 5
ps -p 3361174,3361341,3395333 -o pid=,ppid=,stat=,args=
nvidia-smi
```

进程终止不可恢复，但这些服务可用原命令重新启动；checkpoint、代码和日志没有因 `TERM` 被删除。旧服务退出后 GPU 0 有约 70.8 GiB 可用显存，随后使用已有的 `behavior-server` 会话执行了等价于下列命令的启动：

```bash
tmux new-session -d -s behavior-server \
  -c /run/ti/BEHAVIOR2026/behavior-node-stage1 \
  "/bin/bash -lc 'set -o pipefail; \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    /run/ti/BEHAVIOR2026/behavior/.venv/bin/python \
    scripts/serve_memlite_stage1_behavior.py \
    --root /run/ti/BEHAVIOR2026/memlite-stage1 \
    --port 10110 --device cuda:0 2>&1 | \
    tee /run/ti/BEHAVIOR2026/logs/serve_e2e_10110_20261008.log'"

tmux list-sessions
tmux capture-pane -p -t behavior-server -S -80
```

若 `nvidia-smi` 显示 GPU 0 不满足余量，应只把 `cuda:0` 换成实际空闲卡号，不能在未核对进程归属时抢占其他卡。server 报 ready 后按下文命令执行 health check 和 official synthetic 单请求。

## 现在仍需要的条件

完成正向闭环验收还需要以下三项：

1. 一帧真实 BEHAVIOR evaluator 初始观测，或训练分布中的已审核/缓存 episode 观测。部署包当前只有权重、统计量和资产，没有训练 episode 视频；全零图和物体 diffuse texture 都不能作为正向生成验收输入。
2. 用该真实观测通过 official wire 复验 high AR。若仍在 1024 token 内不到 `<HL_END>`，需要对同一样本做 teacher-forced prefix 与 free generation 对照，定位 high checkpoint 的生成闭合/缓存/解码问题；不能放宽严格 admission 或在 server 中拼接伪 `<HL_END>`。
3. high 通过后，在同一请求中观察 `STAGE1_PLANNER_EVENT` → `STAGE1_LOW_CHUNK` → `(23,)` action，才可把 server 状态改为“跑通”。

当前不需要重新下载模型、修改 low checkpoint或增加 CUDA 依赖。若使用真实 official evaluator 直接产生观测，则仍需要可运行的 OmniGibson graphics 环境；若能提供缓存的真实 wire observation，则 policy server 验收本身不需要 Vulkan。

## 启动步骤

先在正常 Pod shell 检查 CUDA、显存和端口：

```bash
nvidia-smi

/run/ti/BEHAVIOR2026/behavior/.venv/bin/python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.device_count())'

ss -ltnp | rg ':10110'
```

确认目标 GPU 有足够余量后，从新 worktree 启动。将 `<GPU>` 替换为实际空闲卡号：

```bash
cd /run/ti/BEHAVIOR2026/behavior-node-stage1

set -o pipefail
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /run/ti/BEHAVIOR2026/behavior/.venv/bin/python \
  scripts/serve_memlite_stage1_behavior.py \
  --root /run/ti/BEHAVIOR2026/memlite-stage1 \
  --port 10110 \
  --device cuda:<GPU> 2>&1 | \
  tee /run/ti/BEHAVIOR2026/logs/serve_e2e_10110_20261008.log
```

这里不需要显式传 `--max_new_tokens 1024`；服务会从模型配置读取审核预算。如果显式传入其他值，服务会在启动阶段拒绝配置漂移。

双 checkpoint 加载完成后，先检查健康端点：

```bash
curl --fail --max-time 5 http://127.0.0.1:10110/healthz
```

然后可在第二个 shell 中发送一帧 synthetic official observation。这个命令现在只作为握手、严格失败和异常可观测性测试；全零图像不能作为正向闭环通过条件：

```bash
cd /run/ti/BEHAVIOR2026/behavior-node-stage1

/run/ti/BEHAVIOR2026/behavior/.venv/bin/python \
  scripts/smoke_official_client.py \
  --host 127.0.0.1 \
  --port 10110 \
  --task-id 0 \
  --requests 1 \
  --timeout 300 | \
  tee /run/ti/BEHAVIOR2026/logs/smoke_official_10110_20261008.log
```

## Server 跑通标准

以下条件必须同时满足：

- `/healthz` 返回 `OK`。
- 客户端握手为 `policy=G0.5`、`embodiment=R1Pro`、`action_dim=23`、`action_steps=16`。
- server 日志出现 `STAGE1_PLANNER_EVENT`，生成结果到达 `<HL_END>`，没有 ModuleNotFound、target-free validation 或 token-budget 错误。
- 随后出现 `STAGE1_LOW_CHUNK`。
- 最后出现 `STAGE1_ACTION`，客户端收到 shape 为 `(23,)` 且全部有限的 float32 动作。
- 单次测试结束后检查 GPU 使用和进程归属，只停止本次新启动的 10110 服务，不处理旧服务。

Synthetic observation 是全零图像和本体状态。若它成功，只能证明代码、权重和协议链路在该输入上连通；若 high planner 像本次一样严格拒绝，也不能据此单独断言真实 evaluator 输入必然失败。正向验收必须使用真实或已审核缓存观测。

## 常见失败与处理方向

| 失败表现 | 最可能原因 | 处理方向 |
| --- | --- | --- |
| `cudaGetDeviceCount`、`0 active drivers` | 仍在受限沙箱或 GPU device 未挂载 | 换到正常 Pod shell；不要继续用 CPU 强制加载 |
| `Address already in use` | 端口冲突 | 保留旧进程，改用另一个新端口 |
| `No module named g05.utils.memlite_planner_format` | 从旧 worktree 启动 | 核对 cwd、脚本绝对路径和 Git commit 是否为 `78dcc61` |
| `planner max_new_tokens` 不一致 | 显式传入了旧值 160 | 删除该参数，或传 checkpoint 配置值 1024 |
| `Operation creation failed` 或 CuTe JIT 错误 | 实际导入了未适配的旧 `vision.py` | 核对 `g05` 的实际 import 路径必须来自 `behavior-node-stage1/src` |
| Target-free 字段或 template 校验失败 | 启动了旧 server 脚本 | 核对脚本来自新 worktree，并保存完整 traceback |
| `planner AR response hit generation bound before <HL_END>` | high 自由生成不闭合；本次两个非真实输入均已复现 | 先用真实 evaluator/cached episode 复验；仍失败则做同样本 teacher-forced/free-generation 对照，不拼接伪结束符 |
| Planner 成功但没有 low chunk | low processor、normalizer 或动作后处理失败 | 保存 `STAGE1_PLANNER_EVENT` 后的首个 traceback，不降级返回伪动作 |

## Server 通过后仍需完成的事项

Server 单请求通过后还有三项工作：

1. 由另一成员独立 review 本次适配，再 push feature 分支或按团队流程合入部署分支。远端 `deploy/memlite-stage1` 已在 2026-10-08 重新 fetch 并确认仍为本分支基线 `018cce3`。
2. 若目标是完整 BEHAVIOR rollout，还需要 graphics-capable OmniGibson 节点，或让平台为仿真 Pod 提供 NVIDIA graphics/Vulkan 能力。该条件与 policy server 的 compute-only 验收相互独立。

## 当前最短路径

当前最短路径是：保持现有 `behavior-server` / 10110 运行 → 提供一帧真实 evaluator 或缓存 episode observation → 复验 high 是否到 `<HL_END>` → 若通过则收集同一请求的 planner/low/23D action 证据；若仍失败，则对该真实样本做 high teacher-forced/free-generation 对照。

在真实输入复验前，不应凭 synthetic OOD 失败直接重训，也不应修改 strict admission 或伪造 planner 事件。
