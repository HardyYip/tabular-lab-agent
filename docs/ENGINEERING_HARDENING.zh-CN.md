# 截止控制、候选反馈与集成审计

本轮完成三项工程改进，验证使用 HTTP mock、小型合成数据和受限 Docker/GPU 测试。
没有恢复付费 API，也没有重跑或续跑原 Wine Quality benchmark。新源码与旧冻结协议不同。

## 1. 谁决定还能不能继续搜索

`runner.run_task()` 先划定搜索截止时间，把最终训练的预留留在搜索之外。
`JSONClient._window()` 再检查：剩余搜索时间必须容得下完整配置的请求窗口和单候选训练窗口。
凭据读取与 HTTP 客户端创建后再检查一次，避免使用已经过期的时间快照。

例如请求 60 秒、候选训练 120 秒，需要至少 180 秒搜索余额；仅剩 3 秒时正常停止搜索。
没有 HTTP 请求就不增加 API 调用数，也不制造未知用量。已有最佳模型仍进入最终化。
请求返回后还会重新检查训练时间，避免启动被迫缩短的候选训练。

代价是搜索可能更早结束。这里采用已有配置的完整窗口，没有根据测试分数挑选阈值。
HTTPX 的阶段 timeout 不等于整个请求的总耗时硬限，所以仍保留响应后的检查与子进程超时，
不声称整个运行严格不超过 max_seconds。详见 [请求计时](REQUEST_TIMING.zh-CN.md)。

阅读顺序：`runner.py` 的 `search_deadline` → `llm.py` 的 `_window()` → `trial_time_stop`。
对应测试：`test_deadline_requires_full_request_and_training_window` 和
`test_response_overrun_does_not_start_a_shortened_candidate`。

## 2. Agent 选择方案，执行器持有定义

V1 的 `candidate_options` 只包含未尝试的注册候选，每项提供已有 fingerprint 作为
`candidate_id` 和对应的 `experiment`。LLM 返回 ID 和本次假设，执行器查自己的注册表获得参数。
没有扩大候选空间，也没有把越界参数偷偷改成合法参数。旧的完整 ExperimentSpec 格式仍支持 replay，
但仍受原有范围和重复检查约束。提示词只发送一份候选定义，避免重复整张候选表。

拒绝原因通过最近最多三条 `planning_feedback` 进入下一次请求，成功推进后清空。
连续三次非法仍会停止；没有未尝试候选时，在调用 LLM 前结束搜索。

阅读顺序：`llm.Plan` → `runner` 构造 options → candidate_id 查表 → `except ValueError`。
测试 `test_rejected_plans_get_feedback_then_canonical_id_choice_succeeds` 先模拟越界和重复，
再用 ID 纠正，检查成功候选从后续选项移除，而且上下文没有测试信息。

## 3. 如何审计一棵模型对象图

`runtime.inspect_model()` 在容器中沿受支持的组合关系遍历，宿主只接收有界、严格校验的 JSON。
现在支持自定义 `members_` 和受支持的 `estimators_` 列表、元组、字典，成员可继续嵌套包装器和 helper。

| 情况 | 处理 |
| --- | --- |
| 同一对象被多个位置引用 | 去重 |
| 引用环、深度超过 16、对象超过 64、集合超过 64 项 | 审计不完整，设备 unknown |
| 未知成员、异常属性、不支持的集合 | 保留 unknown |
| CPU/GPU 成员同时存在且证据完整 | mixed |
| helper 的已拟合 model_ | 记录对应内部早停证据 |
| helper 同级其他 CatBoost | 独立检查，不借用 helper 的验证状态 |
| helper 元数据损坏 | 保持不完整，仍检查可识别子模型的未验证早停 |

sklearn 原生树集成保留已知 CPU 证据，不把几百棵树逐一扩张。
worker 发现嵌套成员中可识别、未经验证的 CatBoost 早停时也会拒绝。
这是约定属性的受控审计，不是任意 Python 代码的完整证明。

阅读顺序：`runtime.inspect_model()` 的 walk → `RuntimeReport` → `docker/worker.py` 的拒绝条件。
先看 `tests/test_runtime.py` 的环、上限和异常元数据测试，再看
`test_container_audits_custom_members_and_nested_stopping` 的真实容器验证：
正常集成可训练、重训、导出和预测；混入未经验证的成员会被拒绝。

## 交付与复现边界

结果见 [验证状态](VALIDATION.md)。本机 JUnit 保存在被 Git 忽略的
`runs/engineering-hardening-20261027/`。容器测试可用 `TABULAR_LAB_TEST_IMAGE` 选择独立镜像，
GPU 测试仍需 `TABULAR_LAB_GPU_TESTS=1`；临时变量在验证结束后恢复。

最终同一次测试共 103 项通过、无跳过：93 项离线、10 项实际 Docker（其中 2 项 GPU）。
本机默认 worker 标签已指向验证过的新镜像，旧镜像另有 comparison-20261027 保留标签，
已有模型依旧通过原不可变 image ID 加载。

旧对照仍只有 seed 17 两次 partial、两个已导出模型和一次未知 API 用量。
新版不能直接作为旧冻结协议的剩余四次运行补入。实际容器回归不等于真实 LLM 效果实验，
本机测试也不等于新版本已提交、发布或通过远端 CI。
