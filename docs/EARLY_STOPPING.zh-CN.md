# 实际设备记录与内部早停

本轮共同确定的规则：内部监控集选择最佳轮数，随后使用完整的当前训练数据重训。
内部划分默认约 80% 拟合、20% 监控，重复特征行按组划分时比例可能略有偏差。

## 数据流

```text
当前 fit(X, y) 的数据
    ├── 内部拟合部分 → 拟合预处理 → CatBoost 拟合
    └── 内部监控部分 → 使用上述预处理 → 监控指标、选择轮数
                                    ↓
                  新的预处理在完整 X 上拟合
                                    ↓
                  按最佳轮数在完整 X、y 上重训
                                    ↓
                  固定执行器用外部验证集评分
```

预处理也必须在内部拟合部分学习。例如，填补缺失值使用的中位数不能先看内部监控部分。
完整重训时重新拟合预处理，并固定第一阶段实际使用的学习率。
内部默认监控 CatBoost 的分类损失；外部仍按任务指定的 macro-F1 等指标选择方案。
有内部监控不保证一定提前停止；若持续改善，允许训练到轮数上限。

## 四个连接点

1. `tabular_lab/early_stopping.py` 的 `EarlyStoppingCatBoost.fit` 拥有内部划分、预处理和两段训练。
2. `docker/worker.py` 在固定容器执行流程中调用 `fit`，通过 `runtime.inspect_model` 审计已拟合模型。
3. `sandbox.py` 仅在宿主读取经过大小限制和 Pydantic 校验的 `runtime.json`，不加载生成模型。
4. `runner.py` 将运行元数据加入历史反馈、实验报告和导出模型元数据。

生成代码可直接返回：

```python
from tabular_lab.early_stopping import EarlyStoppingCatBoost

def build_pipeline(seed, feature_schema):
    return EarlyStoppingCatBoost(
        params={"iterations": 600, "depth": 6, "thread_count": 4},
        validation_fraction=0.2, patience=50, random_state=seed,
    )
```

GPU 可用时可在 `params` 中加入 `task_type="GPU"`、`devices="0"`、`gpu_ram_part=0.25`。
可选的 `preprocessor` 必须传进组件内部，不能先在整个训练集上拟合再包裹组件。
固定执行器会拒绝检测到的、未通过该组件的 CatBoost 早停，返回 `UnverifiedEarlyStoppingError`。
不使用早停的固定轮数 CatBoost 仍可执行。本轮不强制其他模型更换它们的训练机制。

两段训练与最终流程均受原有墙钟超时控制；不额外赠送训练时间，也不在组件内自动转 CPU。
最终选定方案用训练加外部验证数据拟合时，会在合并后的数据内部重新执行同样的流程；测试标签始终留在外部。

## 看哪些记录

- `device_used`：根据已拟合 CatBoost 参数、受支持的 sklearn CPU 模型及组合关系得出。
  无法识别时记录 `unknown`；同时含 CPU/GPU 的受支持组合记录 `mixed`。
- `runtime.estimators`：设备判断对应的模型路径、证据来源和实际树数量。
- `runtime.early_stopping`：内部两部分大小、尝试轮数、最佳轮数、完整重训行数及各阶段耗时。
- `model/runtime.json`：实际导出模型的记录。最终重训失败时对应保留下来的候选模型。

开放 GPU 不代表实际使用 GPU；记录 CPU 也不自动证明发生了回退，因为 Agent 本来就可以选择 CPU 模型。
老镜像不产生运行元数据时记录 `unknown`，不会凭容器 GPU 开关猜测实际设备。
修改组件后需要重新构建工作镜像；已有模型仍按保存的镜像 ID 加载。

自定义组合现支持沿 `members_` 和受支持的 `estimators_` 列表、元组或字典遍历；
共享对象去重，循环、异常属性、未知节点及超出上限均保持不完整、设备 `unknown`。
最多检查 64 个对象、16 层嵌套和每集合 64 项。sklearn 原生树集成保留已知 CPU 证据。
helper 的早停验证只传给它自己的 `model_`，不会把同级其他模型一起视为已验证。
因此成员中可识别的未经验证 CatBoost 早停也会被 worker 拒绝。
这是约定属性的受控审计，不是对任意 Python 封装的完整证明。

本轮设计依据 CatBoost 的 [fit 接口](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit)
和 [最佳迭代查询接口](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_get_best_iteration)。

## 本机真实验证：2026-10-27

以下为早停阶段的历史记录；最新工作区的统一回归结果见 [验证状态](VALIDATION.md)。

离线回归 62 项通过；补充学习率固定规则后相关 7 项再验通过；另补充 1 项最终重训失败时
导出候选设备记录的测试，通过。实际 Docker 回归 8 项通过，包含 CPU/GPU、拒绝自监控早停、
反馈传递、导出与新进程预测。这些是本机结果，新版 GitHub CI 尚未执行。

随后运行 Wine Quality、seed 42 的真实 DeepSeek 学习实验。2 次 API 调用，估算费用
0.0223776 元，运行约 103.75 秒。两份生成方案都采用组件并使用 GPU：

| 候选 | 轮数上限 | 实际监控轮数 | 最佳轮数／完整重训轮数 | 验证 macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| 第一份 | 1500 | 465 | 385 | 0.562233 |
| 第二份 | 2000 | 794 | 694 | 0.570089 |

每份候选都将 3192 条当前训练数据分为内部拟合 2553 条、监控 639 条，随后按最佳轮数
在完整 3192 条上重训。第二次真实请求中已确认包含第一份方案的设备与早停记录。
选择第二份方案后，在训练加验证的 4256 条数据内部重新选出 900 轮，再完整重训。
最终测试 macro-F1 为 0.614712；新进程的 1064 条预测和概率与导出前完全一致。

代码审查还保留了一个不一致：第二份自然语言假设提及类别权重，实际参数并未设置
`class_weights` 或 `auto_class_weights`。没有替 Agent 修改输出，不把该描述当作已执行事实。
本轮验证组件和反馈闭环，不据此声称早停或 V2 具有普遍效果优势。

本机记录：`runs/early-stopping-agent-learning-20261027/results/20261027T070414Z-e777a92b/`。
`learning-review.json` 包含核对结果，`model/runtime.json` 对应实际导出的模型。
