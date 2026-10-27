# Benchmark 的安全恢复与结果保护

现在默认拒绝写入非空输出目录。`--resume` 只能继续同一协议下尚未开始的运行，不会从头覆盖汇总。
当前真实 API 和旧 Wine Quality benchmark 仍暂停；恢复能力不代表自动授权付费。

## 先看计划，不执行

```powershell
.\.venv\Scripts\python.exe -m tabular_lab benchmark --suite configs/benchmark-v1-v2.yaml --dry-run
```

这个命令只解析配置，展示顺序、策略、种子、输出目录和配置费用上限，不读取密钥、不调用 API，
不启动训练，也不创建结果目录。它不核对服务商余额或当前价格。
任务、种子、策略不能重复；种子要求非负整数，整个 suite 最多 1000 个运行。

## 什么叫同一协议

开始前写入 protocol.json，记录：

- 包源码摘要与 Python／系统／依赖版本。
- 任务配置、运行顺序、数据文件和图片资源指纹。
- V2 使用的不可变 worker 镜像 ID。

恢复时逐项比较；每次进入真正的 run_task 时还重新检查源码、输入与镜像，防止它们在计划后改变。
这不会固定所有硬件和调度状态，也不会消除 GPU 数值差异。
没有该协议的旧结果目录不会自动迁移或续跑；新版不能悄悄混入旧冻结比较。

## 中断后分别怎么处理

| 保存状态 | --resume 行为 |
| --- | --- |
| 已有已核对的汇总行和最终 summary | 跳过，不重新训练或调用 API |
| 最终 summary 已写完，汇总还没写 | 核对配置、源码、输入和镜像后恢复这一行 |
| 运行目录存在但没有最终 summary | 停止，要求先检查用量；不自动重试 |
| 任意运行有 usage_unknown | 继续保持整个 suite 停止 |
| 数据／图片、源码、依赖、配置、顺序或镜像改变 | 拒绝恢复，使用独立的新协议 |
| 同时有另一个进程写入同目录 | 非阻塞锁拒绝第二个写入者 |

实际恢复是在确认协议和费用状态后，给原 benchmark 命令加 `--resume`。
它会执行尚未开始的真实运行，可能调用收费 API；本轮未执行这一操作。
保留下来的 partial／失败运行不会为了分数被自动重跑。

## 存储与计时

运行使用确定的 run-0000、run-0001 目录。results.json 是原子更新的权威检查点；
results.csv 和 aggregate.csv 通过临时文件替换生成，恢复成功后可重建。
恢复完整孤立运行时没有当时的外层墙钟测量，wall_seconds 为 null，wall_seconds_measured 为 false，
不会把内部 seconds 冒充外层墙钟时间。

本机锁文件 .benchmark.lock 可以保留；锁由操作系统管理，进程退出后释放。
不要把删除锁文件当作强行并行写入的办法。这是本地文件锁，不是分布式任务调度器。

源码入口：benchmark._plans → _protocol → _lock → run_benchmark → _saved_summary／_save。
测试覆盖中断恢复、孤立运行、未知用量、数据变化、篡改汇总、并发锁和只读计划。
