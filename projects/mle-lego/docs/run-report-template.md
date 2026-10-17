# 实验运行报告模板

运行结束后，将此模板复制到对应的 GitHub Issue 中。只报告观察到的事实并提供链接；不要在执行报告中解释某方法是否更优。

```text
issue:
objective:
commit:
profile_revision_or_hash:
protocol_hash:
data_source_revision:
data_manifest_and_sha256:
task_model_id_and_revision:
decision_model_id_and_revision:
gpu:
mode:
seeds:
batches:
train_group_budget:
candidate_training_budget:
evaluator_budget:
output_root:

status:
completed_runs:
completed_batches:
accepted:
rejected:
rejection_reasons:
rollbacks:
duplicate_configurations:
invalid_proposals:
command_retries:
resume_count:
final_test_used:
audit_status:

smoke_log:
formal_log:
summary_files:
evidence_archive:
notes:
```

必须附上或链接运行摘要、进度/状态文件、日志、预检报告、审计结果，以及证据归档的校验和。数据集、模型权重、检查点、凭据和私有配置不得放入公开仓库。
