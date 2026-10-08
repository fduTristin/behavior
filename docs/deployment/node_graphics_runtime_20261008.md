# 本节点 NVIDIA 图形运行时适配记录

日期：2026-10-08（北京时间）

节点：`/run/ti/BEHAVIOR2026`

分支：`fix/node-stage1-e2e-20261008`

## 结论

本节点已通过隔离的用户态 NVIDIA 图形运行时恢复 Vulkan/RTX，不需要修改系统目录，也不需要 `/dev/nvidia-modeset` 或 `/dev/nvidia-caps`。实测结果如下：

- Vulkan 1.3 loader 可枚举 8 张 NVIDIA GPU，驱动均为 `580.95.05`。
- PyTorch/CUDA 仍可枚举 8 张 GPU，没有被图形库覆盖。
- GPU1 上的 OmniGibson 3.9.3 / Isaac Sim 5.1 已完成 `launch -> play -> physics step -> RTX render -> shutdown`，探针退出码为 0。
- 最终 Kit 日志未出现 `ERROR_INCOMPATIBLE_DRIVER`、`Graphics plugins not available`、`No CUDA devices found` 或 `no GPU foundation`。
- 探针退出后 GPU1 回落到 4 MiB；GPU0 上现有 `behavior-server` 保持运行，`/healthz` 返回 `OK`。

这修正了 2026-10-07 报告中“必须由平台开放 graphics device capability、容器内无解”的判断。真正缺失的是一组完整且版本一致的 NVIDIA 用户态图形库；只解包 `libnvidia-gl-580` 不足以构成完整运行时。

## 运行时位置与来源

运行时位于：

```text
/run/ti/BEHAVIOR2026/infra/nv580/runtime
```

它不纳入 Git，当前约 892 MiB。核心库来自本机已有的 `libnvidia-gl-580_580.95.05`、`libvulkan1`、`libegl1` 包，并补入 `10.0.0.11` 上已验证的同版本 NVIDIA `.run` 图形共享库。两台机器的 kernel driver 均为 `580.95.05`，抽查的 `libGLX_nvidia`、`libnvidia-glvkspirv`、`libnvidia-rtcore` SHA256 一致。

`runtime/graphics-lib` 只暴露与 `10.0.0.11` 相同的 27 个图形库链接；`runtime/support-lib` 仅暴露 Vulkan loader、Wayland 和 GBM 依赖。不要把整个 `runtime/usr/lib/x86_64-linux-gnu` 加入 `LD_LIBRARY_PATH`，其中的 `libcuda` 会覆盖容器原有 CUDA 驱动库并导致 PhysX 报 `No CUDA devices found`。

## 使用方法

所有需要 Vulkan/RTX 的命令通过包装脚本运行：

```bash
cd /run/ti/BEHAVIOR2026/behavior-node-stage1

scripts/with_node_graphics_runtime.sh \
  env OMNIGIBSON_HEADLESS=1 OMNIGIBSON_GPU_ID=1 \
  /root/miniconda3/envs/behavior/bin/python <command-or-script>
```

包装脚本会检查运行时文件和 kernel driver 版本，并仅为子进程设置 `LD_LIBRARY_PATH` 与 `VK_ICD_FILENAMES`，不会污染当前 shell 或系统动态链接配置。

Vulkan 自检：

```bash
scripts/with_node_graphics_runtime.sh \
  /run/ti/BEHAVIOR2026/infra/nv580/tools/vulkaninfo --summary
```

## 验证证据

最终 OmniGibson/Kit 日志：

```text
/run/ti/BEHAVIOR2026/BEHAVIOR-1K/OmniGibson/appdata/local/logs/Kit/OmniGibson/3.9/kit_20261008_143850.log
```

本轮只验证了空场景启动、PhysX step 和 RTX render。完整 `turning_on_radio` 场景加载、真实 evaluator observation、high planner `<HL_END>`、low FM 和 23 维 action 的全链路 rollout 尚未执行，因此不能据此报告完整任务成功率或闭环策略通过。
