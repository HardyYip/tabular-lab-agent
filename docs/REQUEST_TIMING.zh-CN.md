# 一次请求为什么只等了约三秒

当前实现会在请求前重新检查截止时间，保留完整请求和单候选训练窗口。
本轮只使用离线／容器测试，没有恢复付费调用或真实 benchmark。

## 先沿模块看时间如何传递

`runner.run_task()` 用单调时钟计算剩余运行时间，再为最终训练预留
`min(final_seconds, max_seconds / 3)` 秒，余下的是本轮搜索可用时间 `available`。
旧循环只在 `available <= 1` 时停止，曾可能把一个很短的时间快照传给 HTTP 客户端。
现在 `runner` 传入绝对的 `search_deadline` 和完整单候选训练预留，`JSONClient._window()`
在开始规划和凭据／客户端准备完成后分别重新读取单调时钟。

真实请求只有在 `剩余搜索时间 >= llm.timeout_seconds + trial_seconds` 时才允许发出。
例如配置请求 60 秒、候选训练 120 秒，需要至少 180 秒搜索时间；最终训练预留已在搜索截止时间外扣除。
不足时记录 `planning_time_stop`，不发请求、不增加调用数、不标记未知用量，并保留最佳模型的最终化流程。
这可能提前结束搜索，是减少尾部无用调用的保守选择，并不保证效果更好。

replay 不需要网络窗口，但仍预留完整候选训练时间。显式直接调用 `JSONClient.ask(timeout=3)`
仍可设置较短 HTTP 参数；它不代表 runner 的新准入策略。零或负窗口不会回退到默认 60 秒。
返回方案后还会再次检查训练窗口；如果网络耗时超出预期，不启动被迫缩短的候选训练。

HTTPX 分别限制连接、读取、写入和连接池等待；浮点 timeout 不是整次请求的总墙钟硬限。
`ReadTimeout` 表示等待接收数据超时。参见 [HTTPX 官方说明](https://www.python-httpx.org/advanced/timeouts/)。

## 新日志分别回答什么问题

| 事件／字段 | 含义 |
| --- | --- |
| `llm_request.available_seconds` | 传入 deadline 时为发请求前重算的搜索余额；直接传 timeout 时为调用方快照，未提供则为 null |
| `llm_request.trial_reserve_seconds` | 给返回方案预留的完整训练窗口 |
| `llm_request.configured_timeout_seconds` | 任务配置的请求 timeout 上限 |
| `llm_request.timeout_seconds` | 实际传给 HTTPX 的浮点参数 |
| `llm_response.elapsed_seconds` | 从 HTTP 调用前到响应解析与用量校验后的单调时钟耗时 |
| `llm_usage_unknown.elapsed_seconds` | 从 HTTP 调用前到捕获失败的耗时 |
| `llm_usage_unknown.http_status` | 已收到响应时的状态码；读取超时且未获得响应对象时为 null |

成功与失败事件通过 `call` 对应请求。耗时不包含密钥读取、费用预留、客户端创建或关闭。
新增字段不进入 LLM 提示词；日志仍不写入密钥、请求头或原始错误响应。
replay 读取旧日志时不要求新增字段，也不伪造原调用的网络耗时。

请求失败且无法取得可靠用量时继续标记 `usage_unknown`，禁止后续请求。
已知费用为 0 只表示没有拿到可计费的 token 记录，不表示这次未知请求免费。
收到 HTTP 错误状态同样保留未知用量规则。

## 离线复现及其边界

`tests/test_llm.py` 用 `httpx.MockTransport` 检查实际 HTTP 参数与日志一致、成功和失败耗时、
HTTP 错误状态、准入边界、凭据读取后再次检查，以及未知用量禁止重试。

`tests/test_runner.py` 的 `test_short_search_window_stops_before_request_and_preserves_export`
分别覆盖 V1 agent 与 V2 codegen 的规划分支：

1. 使用小型合成数据，总时限 120 秒、最终训练预留 30 秒。
2. 基线在本机真实训练；模拟时钟将两个基线阶段的时间各推进 43.5 秒。
3. 剩余搜索窗口为 `120 - 30 - 87 = 3` 秒，小于所需的 `60 + 30 = 90` 秒。
4. 验证没有 HTTP 请求、没有未知用量；正常停止搜索并成功导出模型。
5. `tests/test_planning.py` 另模拟响应耗时过长，检查不会启动缩短训练窗口的候选。

测试没有真实 API 调用，codegen 的 Docker preflight 也被替换；没有执行生成代码或真实容器。
MockTransport 不负责模拟网络 timeout 的自动触发，因此单独的超时测试主动抛出异常。
这些测试验证控制流，不证明历史请求必然由同一原因触发。

## 限制和后续学习

本轮选择完整配置窗口，未新增任意固定秒数。后续可以研究依据历史耗时调整窗口，
但不能根据测试分数挑选门槛。HTTP 阶段 timeout 仍不是整个请求的总墙钟硬限；
准入检查、响应后的检查和子进程训练超时共同减少超支，不能声称严格保证整个运行不越过 max_seconds。
要获得 HTTP 总耗时硬限，需要单独设计可中止执行和未知用量处理。

原 Wine Quality 对照结果、冻结协议及未知费用不作改写。修改包源码后，后续实验属于新版本，
不能作为旧协议下的剩余运行悄悄补入。付费 API 和真实 benchmark 保持暂停。
