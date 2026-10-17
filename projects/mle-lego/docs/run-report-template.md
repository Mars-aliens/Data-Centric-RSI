# Experiment Run Report

Copy this template into the assigned GitHub Issue after the run. Report
observed facts and links; do not interpret method superiority in the execution
report.

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

Required attachments or links are the run summaries, progress/state files,
logs, preflight report, audit output, and the checksum of the evidence
archive. Keep datasets, model weights, checkpoints, credentials, and private
profiles outside the public repository.
