import unittest

from local_experiments.aladdin.requirement_state import RequirementState


class RequirementStateTests(unittest.TestCase):
    def test_versioned_add_edit_confirm_and_reject(self):
        state = RequirementState({"helpfulness": "useful answer"})
        state.apply(round_index=0, action="confirm", requirement_id="helpfulness", text="useful answer")
        state.apply(round_index=1, action="add", requirement_id="correctness", text="factually correct")
        state.apply(round_index=2, action="edit", requirement_id="correctness", text="factually correct and relevant", confirm=True)
        state.apply(round_index=3, action="reject", requirement_id="correctness", text="factually correct and relevant")
        snapshot = state.snapshot()
        self.assertEqual(snapshot["version"], 4)
        self.assertEqual(snapshot["requirements"]["correctness"], "factually correct and relevant")
        self.assertEqual(snapshot["confirmed"], ["helpfulness"])
        self.assertEqual(len(snapshot["events"]), 4)


if __name__ == "__main__":
    unittest.main()
