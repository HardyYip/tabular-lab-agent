# 离线图像、文本与表格混合分类

现在可以用同一个固定执行器处理：普通表格、图像分类、文本分类，以及三者的混合输入。
分类标签由数据提供；划分、验证选模、最终测试和导出预测沿用原流程。
当前视觉入口使用固定图像描述，文本使用字符哈希特征；不是视觉语言大模型、OCR 或目标检测服务。

## 先跑通离线示例

以下在 PowerShell、仓库根目录运行，示例不会调用付费 API，也不下载模型权重：

```powershell
.\.venv\Scripts\python.exe -m tabular_lab prepare --dataset image-demo
.\.venv\Scripts\python.exe -m tabular_lab inspect --task configs/image-demo.yaml
.\.venv\Scripts\python.exe -m tabular_lab run --task configs/image-demo.yaml
.\.venv\Scripts\python.exe -m tabular_lab prepare --dataset multimodal-demo
.\.venv\Scripts\python.exe -m tabular_lab run --task configs/multimodal-demo.yaml
```

第一个例子只根据图片区分圆、方形、三角形。第二个例子混合图片、描述文字、数值和类别列；
其中 Agent 是明确标记的 mock 策略。图像位置、颜色与大小有确定性变化，原始文件保存在被忽略的 data/。
这些是功能学习样例，分数不能代表真实场景的视觉泛化能力，也不证明 LLM 有收益。

`inspect` 只读取本机数据，报告训练画像、划分行数、特征数和数据指纹；不输出测试指标或请求密钥。

## 接入自己的数据

CSV 每行是一条样本，例如：

```csv
image,caption,weight,source,label
images/001.png,rounded boundary,1.2,a,circle
images/002.png,straight edges,1.8,b,square
```

任务配置中明确声明哪些列是什么模态；其他列继续按普通表格处理：

```yaml
data_path: ../data/items.csv
target: label
strategy: baseline
modalities:
  image_columns: [image]
  text_columns: [caption]
  image_size: 16
  text_features: 256
```

图片路径默认相对所在 CSV 的目录；也可配置 image_root，相对任务 YAML 解析。
支持 PNG/JPEG/BMP/WEBP，读取第一帧。空单元格表示缺失图片并生成指示特征；
路径不存在、文件损坏、URL、绝对路径或越过图片根目录都会报错，不会悄悄当作正常样本。
解码前检查字节与像素数，编码前检查 max_dense_cells，文本有逐条字符上限。

## 特征从哪里来，为什么不会提前拟合

| 模态 | 默认编码 |
| --- | --- |
| 每个文本列 | 256 维字符 2–4 gram 哈希，适用于中英文；不学习词表或 IDF |
| 每个图像列 | 409 维：缺失标志、16×16 灰度像素、RGB 直方图、分块方向梯度直方图 |
| 其他列 | 原有数值／类别预处理，仍只在当前训练数据上拟合 |

这两种模态编码对每条样本独立、无拟合状态，读取验证／测试图片不等于用它们拟合归一化或词表。
可学习的填补、缩放和模型仍在训练 pipeline 内；早停 helper 继续管理它自己的内部预处理。
原始图片路径和文本列从模型输入中移除，模型不会靠文件名直接预测标签。

画像只包含训练特征统计。V2 容器获得编码后的特征和训练标签，不挂载图片目录或原始文本；
LLM 也不会收到原始样本。数值特征仍包含样本信息，原有禁网和输出校验边界继续有效。

哈希可能碰撞，固定视觉描述也没有预训练语义能力。复杂自然图像、长文档、跨模态推理需要后续专门的编码器和评估，
不能从合成图形的准确率推断已经具备这些能力。

## 划分与预测

相同解码后图片内容即使换文件名或 PNG 压缩方式，也会在同一外层划分组中。
CSV 和图片字节指纹都进入 manifest；图片改变而 CSV 不变也能使恢复协议拒绝继续。
外部训练／验证／测试文件存在重复图片内容时会报错；这不识别裁剪、近似照片等语义重复。

可选 group_column 可把病人、设备或同一来源的样本放进同一外层组，且该列不会作为特征。
组内样本不会拆开，所以比例可能偏离 60/20/20；不能保持全部类别时会报错。
该字段控制外层划分，早停 helper 当前仍按完整编码特征行处理内部重复，不宣称具备完整的内部主体分组能力。

导出目录会保存原始列结构、编码配置和 feature_encoding_version。预测时无需 API：

```powershell
.\.venv\Scripts\python.exe -m tabular_lab predict --model runs/<run>/model --input data/new.csv --output runs/new-predictions.csv
```

如果新 CSV 与图片不在原相对位置，可传 `--image-root <图片根目录>`。未知编码版本会被拒绝；
旧表格模型没有这些字段时仍使用原预测路径。生成模型仍只在记录的受限容器中加载。

代码顺序：config.Modalities → features.encode_features → data.load_dataset → runner → prediction_frame。
先读 tests/test_multimodal.py 的外层分组和新进程预测测试，再看真实 Docker 的编码特征导出测试。
原理参考 [HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html)
及 [Pillow 图片读取](https://pillow.readthedocs.io/en/stable/reference/Image.html)。
