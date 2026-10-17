from __future__ import annotations

from .models import TaskRecord, TaskSpec
from .registry import TaskRegistry


def build_local_text_registry() -> TaskRegistry:
    """Deterministic support-intent task with a small stress evaluation split."""

    registry = TaskRegistry()
    registry.register_task(
        TaskSpec(
            task_id="local-intent",
            name="Local support intent classification",
            input_schema="short English text",
            target_schema="intent label",
        )
    )
    registry.add_records(
        [
            TaskRecord(
                "seed-001",
                "local-intent",
                "seed",
                "reset my password",
                "account",
                source_group_id="seed-account",
            ),
            TaskRecord(
                "seed-002",
                "local-intent",
                "seed",
                "my account is locked",
                "account",
                source_group_id="seed-account",
            ),
            TaskRecord(
                "seed-003",
                "local-intent",
                "seed",
                "where can I find my invoice",
                "billing",
                source_group_id="seed-billing",
            ),
            TaskRecord(
                "seed-004",
                "local-intent",
                "seed",
                "please send the receipt",
                "billing",
                source_group_id="seed-billing",
            ),
            TaskRecord(
                "dev-001",
                "local-intent",
                "development",
                "I cannot login to my profile",
                "account",
                source_group_id="dev-account",
            ),
            TaskRecord(
                "dev-002",
                "local-intent",
                "development",
                "the payment receipt is missing",
                "billing",
                source_group_id="dev-billing",
            ),
            TaskRecord("dev-003", "local-intent", "development", "my sign-in details stopped working", "account", source_group_id="dev-account"),
            TaskRecord("dev-004", "local-intent", "development", "I am locked out of the member area", "account", source_group_id="dev-account"),
            TaskRecord("dev-005", "local-intent", "development", "can you resend the charge document", "billing", source_group_id="dev-billing"),
            TaskRecord("dev-006", "local-intent", "development", "the order payment confirmation never arrived", "billing", source_group_id="dev-billing"),
            TaskRecord("dev-007", "local-intent", "development", "I need to get back into my profile", "account", source_group_id="dev-account"),
            TaskRecord("dev-008", "local-intent", "development", "where is the statement for this purchase", "billing", source_group_id="dev-billing"),
            TaskRecord("dev-009", "local-intent", "development", "my login is rejected every time", "account", source_group_id="dev-account"),
            TaskRecord("dev-010", "local-intent", "development", "please send proof of the transaction", "billing", source_group_id="dev-billing"),
            TaskRecord(
                "held-001",
                "local-intent",
                "held_out",
                "help me access my user profile",
                "account",
                source_group_id="held-account",
            ),
            TaskRecord(
                "held-002",
                "local-intent",
                "held_out",
                "I need a copy of the payment invoice",
                "billing",
                source_group_id="held-billing",
            ),
            TaskRecord("held-003", "local-intent", "held_out", "the app will not let me sign in", "account", source_group_id="held-account"),
            TaskRecord("held-004", "local-intent", "held_out", "I forgot the code for my account", "account", source_group_id="held-account"),
            TaskRecord("held-005", "local-intent", "held_out", "could I get the bill from last month", "billing", source_group_id="held-billing"),
            TaskRecord("held-006", "local-intent", "held_out", "my purchase paperwork is unavailable", "billing", source_group_id="held-billing"),
            TaskRecord("held-007", "local-intent", "held_out", "why cannot I enter the customer portal", "account", source_group_id="held-account"),
            TaskRecord("held-008", "local-intent", "held_out", "I am looking for the payment record", "billing", source_group_id="held-billing"),
            TaskRecord("held-009", "local-intent", "held_out", "please help restore my access", "account", source_group_id="held-account"),
            TaskRecord("held-010", "local-intent", "held_out", "send me the document for my order", "billing", source_group_id="held-billing"),
            TaskRecord(
                "test-001",
                "local-intent",
                "final_test",
                "my sign in details do not work",
                "account",
                source_group_id="test-account",
            ),
            TaskRecord(
                "test-002",
                "local-intent",
                "final_test",
                "where is the bill for my order",
                "billing",
                source_group_id="test-billing",
            ),
            TaskRecord("test-003", "local-intent", "final_test", "my credentials are no longer accepted", "account", source_group_id="test-account"),
            TaskRecord("test-004", "local-intent", "final_test", "I cannot open the customer area", "account", source_group_id="test-account"),
            TaskRecord("test-005", "local-intent", "final_test", "I need the receipt for my purchase", "billing", source_group_id="test-billing"),
            TaskRecord("test-006", "local-intent", "final_test", "show me my transaction statement", "billing", source_group_id="test-billing"),
            TaskRecord("test-007", "local-intent", "final_test", "the account entry screen keeps failing", "account", source_group_id="test-account"),
            TaskRecord("test-008", "local-intent", "final_test", "can you locate my order record", "billing", source_group_id="test-billing"),
            TaskRecord("test-009", "local-intent", "final_test", "I lost access to the service", "account", source_group_id="test-account"),
            TaskRecord("test-010", "local-intent", "final_test", "I was charged but have no invoice", "billing", source_group_id="test-billing"),
        ]
    )
    return registry
