# Dataset protocol and expansion roadmap

The immediate release is **tabular classification**. Text/image entries below are a roadmap, not
implemented adapters. Sources are public and downloadable; preserve their licenses and attribution.

| Phase | Dataset | Task and evaluation |
| --- | --- | --- |
| V1/V2 | [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) | Red+white, 6,497 original rows; fixed low/medium/high quality classes, macro-F1; deduplication before grouped split |
| V1/V2 | [Adult](https://archive.ics.uci.edu/dataset/2/adult) | Income binary classification; official test retained, development cleaned/split; macro-F1, accuracy, ROC-AUC |
| Text | [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) | Spam classification; normalize/group duplicate messages before stratified splitting; spam F1 and macro-F1 |
| Text | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) | 77 intent classes; official test retained, train/validation split; accuracy and macro-F1 |
| Image | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) | 10 clothing classes; preserve official test, split official training; accuracy and macro-F1 |
| Image | [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) | 37 breeds; preserve published split, training-only validation; pretrained features and lightweight fine-tuning |

Wine quality is ordinal; macro-F1 treats the fixed classes as nominal and does not measure how far a
mistake is from the true quality. Original-score classification is a future separate task preset,
not a label transformation selected after seeing test results. Removing exact duplicate original rows
retains 5,320 records in the downloaded version; feature-group splitting also keeps collisions with
different original scores together. Never keep the original quality score as an input feature.

Adult preparation uses only test **features**, not labels, to remove overlapping development vectors.
The official 16,281-row test remains unchanged. The cleaned development set is split once with seed 42;
benchmark seeds change models/search, not that fixed external validation split. This cleaning means
results are not strictly interchangeable with published unmodified-training benchmarks.

Preparation saves SHA-256 of downloaded archives and run manifests hash the prepared CSV files.
The code license is MIT; Wine Quality and Adult are CC BY 4.0 and need separate source attribution.
Future text/image adapters must verify their own licenses and class/split manifests at implementation.

Small sklearn datasets (`breast-cancer`, `wine`) and a synthetic mixed-type demo remain available for
offline tests and short CI runs. The sklearn Wine cultivar task differs from UCI Wine Quality.
