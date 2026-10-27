# V2 的 GPU 设备选择

本轮只接通设备配置、Agent 上下文、容器执行和模型导出，不做 V1/V2 效果比较。

## 三种模式

| task.yaml 的 device | 含义 |
| --- | --- |
| cpu | 容器不申请 GPU，Agent 使用 CPU 模型配置。 |
| auto | 在实际工作镜像中检测 GPU；成功则开放 GPU，失败则使用 CPU。 |
| gpu | 要求容器 GPU 检测成功，否则在调用 LLM 前报错。 |

检测通过只代表容器能够访问显卡。生成代码仍可选择逻辑回归、ExtraTrees 等 CPU 模型。
CatBoost 使用 GPU 需要在生成代码中明确设置 `task_type="GPU"`。镜像没有安装 PyTorch。
`auto` 的回退发生在启动检测阶段；训练出错仍进入原来的错误反馈与修复流程，不自动改写源代码。

## 从配置到执行

1. `runner.run_task` 调用 `sandbox.preflight(image, device)`，解析实际可开放的设备。
2. 原始 `auto` 等请求保留在 `config.json`；解析结果单独进入 manifest 和 summary 的 `execution_environment`。
3. `runner` 将 `gpu_enabled`、GPU 名称、CPU 和内存限制放入 Agent 的 `execution_limits`。
4. `llm.code_prompt` 解释这些约束；Agent 据此生成模型配置。
5. `sandbox.docker_command` 只为已解析的 GPU 模式加入 `--gpus=all`。
6. 每个候选和导出模型的 `image.json` 保存设备；最终重训与新进程预测恢复相同设置。

`image.json` 的设备表示容器访问权限，不是对生成模型实际训练设备的测量。
GPU 模式导出的容器模型目前在预测时也要求 GPU 可用。没有设备字段的旧模型继续按 CPU 加载。
预测检查记录的镜像 ID，不要求原来的可变镜像标签仍存在。

## 不付费的检查

PowerShell，仓库根目录：

```powershell
.\.venv\Scripts\python.exe -m tabular_lab docker-check --device auto
```

查看返回的 `requested_device`、`device`、`gpu_enabled` 和 `gpu_devices`。
这个命令不会调用 DeepSeek。`docker.memory` 限制的是主机内存，不是显存；
CatBoost 的 `gpu_ram_part` 也是模型配置建议，不是执行器强制实施的显存配额。

GPU 集成测试使用固定代码和 replay，不调用真实 LLM。需要运行中的 Docker、可用的 GPU 和
`TABULAR_LAB_GPU_TESTS=1` 才会执行；普通无 GPU 的 CI 会跳过这两项 GPU 测试。

## 本机验证记录：2026-10-27

- 静态检查通过；离线回归 55 项通过。
- 实际 Docker 回归 6 项通过，包含 GPU 访问下的 replay 修复流程，以及 CatBoost GPU
  训练、按候选记录恢复 GPU 重训、导出后新进程预测与概率一致性验证。
- 正式 CLI 的 `--device auto` 检测到 RTX 5070 Laptop，并解析为 GPU。
- 本轮没有真实 DeepSeek 调用，不能据此声称 LLM 已自主选择 GPU 或 V2 有效果优势。
- 本轮记录为本机验证；没有提交、推送或运行新版 GitHub CI。

Docker 测试的本机 JUnit 记录在 `runs/gpu-device-tests-20261027.xml`，原始测试产物留在
`runs/gpu-device-tests-20261027-*`。这些包含模型的本机文件不进入公开仓库。

## 后续真实 Agent 学习实验：2026-10-27

使用 Wine Quality、seed 42、`device: auto`，最多执行两个基线和一个生成候选。
本次实际调用 DeepSeek 一次，估算费用 0.01229688 元，运行约 91.9 秒，无失败候选。
计价采用核对过的 [官方峰时、未命中缓存价格](https://api-docs.deepseek.com/quick_start/pricing/)
和配置中的假定汇率 7.20 CNY/USD，不代表服务商实际账单。

| 验证集方案 | macro-F1 | accuracy |
| --- | ---: | ---: |
| Dummy | 0.202443 | 0.436090 |
| 逻辑回归 | 0.547252 | 0.574248 |
| Agent 生成 CatBoost | 0.557297 | 0.570489 |

执行器依据事先指定的 macro-F1 选择生成模型；它的 accuracy 没有超过逻辑回归。
最终重训后的测试 macro-F1 为 0.614740。候选与最终模型在容器内审计的真实
`task_type` 均为 GPU；两者各训练 1500 棵树。新进程的 1064 条预测及概率完全一致。

代码审查发现两个待讨论项，原始输出已保留，没有替 Agent 修改：

- 生成代码捕获任意训练异常后尝试 CPU；仅凭生成的 GPU 配置不能证明实际使用设备。
- `eval_set=(Xall, y)` 使用训练数据本身。它没有接触外部验证标签，但无法提供独立的泛化
  监控信号；本次确实训练满 1500 轮，没有提前停止。

这是一轮功能与教学实验，不是同预算的 V1/V2 效果对照。
本机原始记录：`runs/gpu-agent-learning-20261027/results/20261027T063119Z-d031175e/`。
其中 `learning-review.json` 保存核对结果，`device-audit/actual-devices.json` 保存已拟合参数。
