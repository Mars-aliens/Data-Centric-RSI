# 本地验证记录

验证日期：2026-11-06。仅验证 Aladdin；未修改 MLE-LEGO，未上传 GitHub。

## 已通过

- `python -m pytest -q`：96 passed。
- `python -m compileall -q experiments scripts tests`：通过。
- Git diff 空白检查：通过。
- 六份 RSI 示例配置：全部通过 runner 配置校验。
- 五种比较矩阵（overall、ablation、factorial、selection、all）：使用合成测试输入、两个 seed、两轮，执行生成、清洗、选择、反馈、策略更新、训练调用及分析流程。
- 所有比较矩阵的首轮生成内容共享；overall、ablation、factorial 首轮训练语义多重集相同。
- 在受控测试反馈中，三项策略更新可以触发，被冻结的模块保持不变。
- 完整因子分析：用已知含两两交互和三项交互的响应曲面核验七项效应；按每 seed 最后一轮计算。
- 首轮训练内容不依赖 arm ID 与反馈到达顺序；训练池先无放回使用记录，再补充重复样本。
- 非法模型反馈、恢复重试、API 超时均保留 attempt；完成 query 不重复收集；未知 token 用量标记为不完整。
- 现有数据预处理、清洗规则、人工标注交互、模型/真人一致性分析、final-test 盲评及恢复测试通过。交互测试均为预设测试输入，不是实际真人研究。
- 四个命令行入口的 `--help` 通过：RSI runner、结果分析、反馈一致性比较、模型服务检查。此项仅验证入口可加载，不表示模型服务在线。

## 验证范围

协议集成测试使用模拟生成器、测试反馈和替代训练函数，不读取真实 final-test，不下载模型，也不执行 Qwen GPU 训练。当前 checkout 没有 `data/processed/prism/`，因此生产数据的内容、manifest 和实际分组尚未在本地复核。

仍需服务器实测：PRISM 正式预处理数据核对、Qwen3-8B GPU 训练、Qwen3-32B API 联调、实际训练 token/耗时和真实 API 重试成本。真人清洗审计、反馈一致性和最终盲评需真实参与者，不能用测试输入替代。

## 重跑

在 `projects/aladdin/` 执行：

```bash
python -m pytest -q
python -m compileall -q experiments scripts tests
```

本次完整命令、退出码和日志保存在本地 `D:/RSI/results/aladdin-local-validation-20261106/`；`verification.json` 汇总已执行检查与未验证范围。真实实验须使用新输出目录，不续接旧代码产生的结果。
