# MEMLite Stage 1 闭环 Server 运行状态与剩余条件

- 日期：2026-10-08（北京时间）
- 节点：`/run/ti/BEHAVIOR2026`
- 适配分支：`fix/node-stage1-e2e-20261008`
- 代码提交：`78dcc6145b241d471cd87eb2c5a090ef506794fa`

## 当前结论

闭环 server 的已知代码阻塞已经修复，当前状态是“代码已具备 GPU 冒烟条件，真实 GPU 闭环尚未验收”。现在最优先需要的不是继续改模型或补依赖，而是在能访问 CUDA 设备并允许监听 localhost 端口的正常 Pod shell 中启动新分支，用 synthetic official client 完成一帧真实推理。

2026-10-08 12:20（北京时间）按用户要求尝试清理旧 GPU 服务并在 `behavior-server` tmux session 启动完整服务，但当前 Codex shell 位于独立 PID/network/device namespace：它能通过宿主 `/proc` 列出旧进程，却不能向这些宿主 PID 发信号；也不能连接已有 tmux socket或创建隔离 tmux socket。因此本次没有实际停止任何进程，也没有制造一个无法工作的假 server session。需要从正常 Pod shell 执行下文的“宿主操作交接”。

Policy server 本身只需要 CUDA compute 能力，不需要 Vulkan 或 GPU graphics。旧报告中的 graphics capability 问题只阻塞 OmniGibson 仿真，不阻塞 high planner、low FM 和 WebSocket server 的独立验证。

## 当前代码状态

新 worktree 位于：

```text
/run/ti/BEHAVIOR2026/behavior-node-stage1
```

它基于本地可用的 `deploy/memlite-stage1@018cce3` 创建。原 `/run/ti/BEHAVIOR2026/behavior` 仍被旧的 10050、10051 和 10100 服务进程引用，未切分支、未 pull、未热改。2026-10-08 12:20 的停服尝试受 PID namespace 隔离阻塞，三个旧服务仍在运行。

已完成的本节点适配包括：

- 从本地 Git 对象精确恢复 `src/g05/utils/memlite_planner_format.py`。恢复后的 blob 为 `983692191891662962cda0fcd3ff0541b6977a62`，与残留原始对象一致。
- 支持 recipe 中以标量保存的 state/action `raw_shape`，避免对整数取 `[-1]`。
- 保持官方 bridge 输入的 CHW 图像布局，生成可写的 `[T,C,H,W]` tensor，不再错误转换成 HWC。
- 将 processor 的 canonical 原始字段和 `planner_*` 渲染别名转换为模型要求的 target-free prefix，并把 template 精确截断到 `<EOC>`。
- 在 sm_120 Blackwell 上跳过当前不兼容的 FA4/FA2 wheel，使用 SDPA fallback。
- planner token budget 默认继承 checkpoint recipe 的审核值，不再沿用旧的 160-token 默认值；当前 checkpoint 配置为 1024。
- 新增 `scripts/smoke_official_client.py`，可在不启动 OmniGibson 的情况下验证握手、reset、planner AR、low FM 和 23 维动作响应。

当前分支尚未 push。`git fetch origin --prune` 因本执行环境无法解析 `github.com` 而失败，所以 `018cce3` 是本地可证明的最新 remote-tracking 基线，不代表已经核实 GitHub 此刻没有更新。

## 已完成验证

| 验证项 | 结果 | 结论边界 |
| --- | --- | --- |
| Stage 1 和 MEMLite 相关 CPU 回归 | 52 项通过 | 覆盖 memory、标签、HL end、processor、runtime 和 conditioning 合同 |
| 官方 bridge 非 socket 回归 | 7 项通过 | 覆盖官方观测、动作映射和 msgpack codec；socket 项受当前沙箱限制未执行 |
| 缺失 planner format 模块 | 导入和单测通过 | receipt、batch、`<HL_END>` token 一致性均有回归 |
| 真实 high recipe 和 processor | 通过 | 三相机输出均为 `[1,3,256,256]` |
| Target-free planner prefix | 通过 | 真实 processor 输出经转换后通过模型自身 `_validate_target_free_high_prefix` |
| CLI 和 smoke client | 编译及 `--help` 通过 | 尚未连接真实 GPU server |
| 新分支 high 和 low checkpoint GPU restore | 未执行 | 当前 Codex 沙箱无 CUDA 设备 |
| Planner AR 到 `<HL_END>` | 未执行 | 需要真实 CUDA 推理 |
| Low FM chunk 和 23 维动作响应 | 未执行 | 需要 planner AR 首先成功 |
| OmniGibson rollout | 未执行 | 另受 graphics capability 阻塞 |

当前 Codex 执行沙箱中的实际环境是：没有 `/dev/nvidia*`，`torch.cuda.is_available()` 为 `False`，`cudaGetDeviceCount` 返回 error 304，且 localhost socket 监听受限。强制在 CPU 上构造模型会在 FLA/Triton 初始化时因没有 active CUDA driver 退出。这些结果说明本执行沙箱不能承担最终 GPU 验收，不等于 checkpoint 或适配代码加载失败。

## 2026-10-08 宿主操作交接

12:20 二次核对确认以下三个主进程均属于本项目旧服务：

| PID | 服务 | 设备 | 端口 |
| ---: | --- | --- | ---: |
| `3361174` | `serve_memlite_stage1.py --branch low` | `cuda:0` | `10051` |
| `3361341` | `serve_memlite_stage1.py --branch high` | `cuda:1` | `10050` |
| `3395333` | `serve_memlite_stage1_behavior.py` | `cuda:4` | `10100` |

当前 shell 对每个 PID 执行 `kill -TERM` 都返回 `No such process`，但随后从宿主 `/proc` 仍能看到相同 PID、启动时间和命令行。这是 PID namespace 隔离，不是进程已经退出。三个主进程及其 TorchInductor worker 因而都没有被本次操作停止。

同一 shell 的运行能力预检结果为：

- `tmux list-sessions` 连接 `/tmp/tmux-0/default` 返回 `Operation not permitted`。
- 在 workspace 下尝试创建独立 tmux socket同样返回 `Operation not permitted`。
- 共享运行环境是 PyTorch `2.7.1+cu128`，但 `torch.cuda.is_available()` 为 `False`、device count 为 `0`，并且 `/dev/nvidia*` 不存在。
- 创建 IPv4 socket即返回 `PermissionError: Operation not permitted`。

因此只有正常 Pod shell 能完成以下宿主操作。先逐 PID 复核命令行再温和停止，不能使用 `pkill python`：

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

进程终止不可恢复，但这些服务可用原命令重新启动；checkpoint、代码和日志不会因 `TERM` 被删除。确认旧服务退出、目标 GPU 至少有 30 GiB 可用显存且 `10110` 未监听后，创建用户指定的完整服务会话（下面以释放后的 GPU 0 为例）：

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

## 立即需要的条件

完成 server 验收需要以下四项：

1. 一个正常的 Pod shell，能看到 `/dev/nvidiactl`、目标 `/dev/nvidiaN`，且 `torch.cuda.is_available()` 为 `True`。
2. 一张经 `nvidia-smi` 确认空闲的 GPU。旧实测双模型常驻约 18.8 GiB，建议启动前至少保留 30 GiB 可用显存，不能仅凭卡号假定空闲。
3. 一个未占用的新端口，建议使用 `10110`，避免与旧 10100 服务混淆。
4. 对新分支执行一次 synthetic official 单请求。只有看到 planner、low chunk 和 23 维动作三段证据，才能把 server 状态改为“跑通”。

不需要重新下载模型，不需要改 checkpoint，不需要 Vulkan，也不需要先启动 OmniGibson。

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

然后在第二个正常 shell 中发送一帧 synthetic official observation：

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

Synthetic observation 是全零图像和本体状态，因此生成的技能内容及动作质量没有任务效果含义。它只证明代码、权重、协议和双模型推理链路连通。

## 常见失败与处理方向

| 失败表现 | 最可能原因 | 处理方向 |
| --- | --- | --- |
| `cudaGetDeviceCount`、`0 active drivers` | 仍在受限沙箱或 GPU device 未挂载 | 换到正常 Pod shell；不要继续用 CPU 强制加载 |
| `Address already in use` | 端口冲突 | 保留旧进程，改用另一个新端口 |
| `No module named g05.utils.memlite_planner_format` | 从旧 worktree 启动 | 核对 cwd、脚本绝对路径和 Git commit 是否为 `78dcc61` |
| `planner max_new_tokens` 不一致 | 显式传入了旧值 160 | 删除该参数，或传 checkpoint 配置值 1024 |
| `Operation creation failed` 或 CuTe JIT 错误 | 实际导入了未适配的旧 `vision.py` | 核对 `g05` 的实际 import 路径必须来自 `behavior-node-stage1/src` |
| Target-free 字段或 template 校验失败 | 启动了旧 server 脚本 | 核对脚本来自新 worktree，并保存完整 traceback |
| Planner 成功但没有 low chunk | low processor、normalizer 或动作后处理失败 | 保存 `STAGE1_PLANNER_EVENT` 后的首个 traceback，不降级返回伪动作 |

## Server 通过后仍需完成的事项

Server 单请求通过后还有三项工作：

1. 恢复 GitHub 网络后执行 `git fetch origin --prune`，确认远端 `deploy/memlite-stage1` 是否又有更新。如果远端已前进，应审查差异后把 `78dcc61` 迁移到新基线，不能盲目覆盖。
2. 由另一成员独立 review 本次适配，再 push feature 分支或按团队流程合入部署分支。
3. 若目标是完整 BEHAVIOR rollout，还需要 graphics-capable OmniGibson 节点，或让平台为仿真 Pod 提供 NVIDIA graphics/Vulkan 能力。该条件与 policy server 的 compute-only 验收相互独立。

## 当前最短路径

当前最短路径是：在正常 CUDA shell 中选择空闲 GPU → 从 `behavior-node-stage1@78dcc61` 启动 10110 → 运行一次 `smoke_official_client.py` → 保存 server 日志和客户端 JSON → 根据上述跑通标准更新状态。

在这一步成功前，不需要继续修改训练超参、重训模型或启动 OmniGibson。
