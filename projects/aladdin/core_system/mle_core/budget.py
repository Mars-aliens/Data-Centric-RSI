from __future__ import annotations

from dataclasses import dataclass

from .models import Budget


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetLedger:
    limit: Budget
    used: Budget = Budget()

    def charge(self, amount: Budget) -> None:
        next_used = self.used.add(amount)
        if next_used.exceeds(self.limit):
            raise BudgetExceeded(
                f"Budget exceeded: used={next_used}, limit={self.limit}"
            )
        self.used = next_used

    def remaining(self) -> Budget:
        return Budget(
            llm_calls=max(0, self.limit.llm_calls - self.used.llm_calls),
            llm_tokens=max(0, self.limit.llm_tokens - self.used.llm_tokens),
            training_steps=max(
                0, self.limit.training_steps - self.used.training_steps
            ),
            training_seconds=max(
                0.0, self.limit.training_seconds - self.used.training_seconds
            ),
            evaluator_calls=max(
                0, self.limit.evaluator_calls - self.used.evaluator_calls
            ),
            feedback_requests=max(
                0, self.limit.feedback_requests - self.used.feedback_requests
            ),
            wall_clock_seconds=max(
                0.0, self.limit.wall_clock_seconds - self.used.wall_clock_seconds
            ),
        )
