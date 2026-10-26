# V2 第一次真实代码生成实验

本次用已有 demo 数据验证 DeepSeek 生成代码 → Docker 训练 → 宿主评分 → 反馈的链路。
demo 是合成数据，结果用于功能验证，不用于证明真实业务效果。

## 配置与边界

使用 `configs/codegen-smoke.yaml`，保留原有 `configs/codegen.yaml`。

- 固定随机种子 42，使用 CPU 容器，限制为 4 CPU、8 GB 内存。
- 先运行 Dummy 和逻辑回归两个基线，再允许最多三个生成代码候选。
- 失败后的修复也是一个候选，计入总共五个候选的上限；每个候选最多修复两次。
- Agent 可以提前停止；候选上限不是保证完成的数量，也不是严格的 API 调用次数上限。
- LLM 估算预算 1 元，运行时间预算 300 秒，单次训练和最终重训各最多 60 秒。
- 价格和汇率使用配置快照，费用是估算值；实际账单以提供商为准。
- 时间预算用于限制规划与训练；Docker 检查、清理和文件导出可能产生额外开销。
- 数据画像、列名和实验反馈会发送给 DeepSeek，默认不发送原始数据行。
- 搜索时验证标签留在宿主评分器，测试集只用于最终选定方案的评估。

## 启动

终端类型：Windows PowerShell。运行目录：`D:\Codex\tabular-lab-agent`。

```powershell
Set-Location 'D:\Codex\tabular-lab-agent'
& .\.venv\Scripts\python.exe -m tabular_lab auth login
& .\.venv\Scripts\python.exe -m tabular_lab run --task configs/codegen-smoke.yaml
```

`auth login` 只需首次配置或更换密钥时执行，输入隐藏，保存到当前用户的 Windows 凭据管理器。
关闭终端、重新启动项目后仍可自动读取；以后直接执行最后一行即可。
最后一行会发起真实 API 请求并产生费用；保存密钥本身不调用 API。
密钥仅用于宿主程序调用 API，不写入任务配置，也不传入训练容器。预测不需要密钥。

`auth status` 只检查本机是否已配置，不显示密钥、不验证 API；`auth logout` 删除保存的密钥。
显式设置的 `DEEPSEEK_API_KEY` 环境变量优先于本机凭据，删除保存的密钥不会清除环境变量。
Linux/CI 继续使用环境变量。Windows 凭据允许同一用户下的程序读取，不是对同账号进程的隔离。

## 如何验收

命令结束后会输出运行目录。先检查其中的 `summary.json`、`experiments.json` 和 `events.jsonl`。

1. 配置为 `strategy: codegen`、`provider: deepseek`，并有真实 API 响应和 usage 记录。
2. 至少一个 `spec.model` 为 `generated` 的候选 `status` 为 `ok`，有验证指标。
3. 对应 `trials/<编号>/container-input/candidate.py` 保存了生成代码，容器输出保存了预测数组和模型。
4. 检查最终模型导出情况，并在新的进程中加载预测；若最佳方案来自基线，明确记录其 native 后端。

整体 `status: completed` 不足以证明 V2 成功：即使所有生成候选都失败，基线仍可能成功导出。
生成候选运行成功也不要求它超过基线。最终选择仍遵守验证指标，不能为了展示 V2 而强行选择生成模型。

如果发生错误，查看下一次 `llm_request` 是否带有 `repair`，以及后续候选是否修复成功。
如果没有发生错误，本次只能验证正常路径，不能声称已经验证真实 LLM 修复。

## 一起阅读的调用关系

`runner.run_task()` 组织上下文 → `llm.JSONClient.ask()` 请求方案 → `CodePlan` 校验结构 →
`sandbox.evaluate_code()` 检查代码并运行容器 → `docker/worker.py` 训练和预测 →
宿主读取数值数组并评分 → `runner` 把分数或错误带入下一轮。

先一起阅读一个成功候选的实验假设、`build_pipeline()` 和验证分数，
再检查下一次请求实际收到了什么反馈。暂不增加新模型或扩大实验规模。
