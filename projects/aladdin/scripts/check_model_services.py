"""检查 Aladdin 生成与模拟反馈服务，不读取实验数据。"""

import argparse
import json
import os
from pathlib import Path

from experiments.feedback.model_feedback import feedback_prompt, parse_feedback
from experiments.rsi.runner import _validate_config
from experiments.synthesis.generator import OpenAICompatibleGenerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = _validate_config(json.loads(Path(args.config).read_text(encoding="utf-8")))
    reports = {}
    for role, endpoint, model, key_env, prompt, tokens in (
        ("generation", config["endpoint"], config["model"], config["api_key_env"],
         "Reply with a short greeting.", 32),
        ("feedback", config["feedback_endpoint"], config["feedback_model"], config["feedback_api_key_env"],
         feedback_prompt({"input": "What is 2 + 2?", "output": "4."}), config["feedback_max_tokens"]),
    ):
        client = OpenAICompatibleGenerator(endpoint, model, api_key=os.environ.get(key_env),
                                          timeout_seconds=config["timeout_seconds"])
        response = client.generate(prompt, max_tokens=tokens, temperature=0)
        if client.usage.complete_usage_requests != 1:
            raise RuntimeError(f"{role} service did not return complete token usage")
        if role == "feedback":
            parse_feedback(response)
        reports[role] = {"model": model, "prompt_tokens": client.usage.prompt_tokens,
                         "completion_tokens": client.usage.completion_tokens, "status": "ok"}
    print(json.dumps({"status": "passed", "services": reports,
                      "note": "Checks API compatibility only; does not verify weights or judge quality."}, indent=2))


if __name__ == "__main__":
    main()
