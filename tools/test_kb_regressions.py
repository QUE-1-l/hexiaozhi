"""政策问答引用与连续追问的回归测试（仅使用标准库）。

运行：python tools/test_kb_regressions.py
"""

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import kb  # noqa: E402


class PolicyQuestionRegressionTests(unittest.TestCase):
    def test_policy_question_after_unmatched_history_keeps_citations(self):
        history = [
            {"role": "user", "content": "1"},
            {"role": "assistant", "content": kb.answer("1")},
        ]
        cases = {
            "学生证丢了怎么办？": "student-id",
            "怎么申请国家奖学金？": "scholarship",
            "有哪些学生资助政策？": "aid-policies",
            "校园卡丢了怎么补办？": "campus-card",
        }
        for question, intent in cases.items():
            with self.subTest(question=question):
                result = kb.answer(question, history=history)
                self.assertEqual(result["intent"], intent)
                self.assertTrue(result["matched"])
                self.assertFalse(result["context_used"])
                self.assertTrue(result["citations"])

    def test_complete_question_is_not_hijacked_by_previous_answer(self):
        first = kb.answer("怎么申请国家奖学金？")
        history = [
            {"role": "user", "content": "怎么申请国家奖学金？"},
            {"role": "assistant", "content": first},
        ]
        result = kb.answer("学生证丢了怎么办？", history=history)
        self.assertEqual(result["intent"], "student-id")
        self.assertFalse(result["context_used"])
        self.assertEqual(result["citations"][0]["doc_id"], "doc-student-id")

    def test_short_followup_still_uses_previous_policy(self):
        first = kb.answer("学生证丢了怎么办？")
        history = [
            {"role": "user", "content": "学生证丢了怎么办？"},
            {"role": "assistant", "content": first},
        ]
        result = kb.answer("那要带什么材料？", history=history)
        self.assertEqual(result["intent"], "student-id")
        self.assertTrue(result["context_used"])
        self.assertTrue(result["citations"])

    def test_all_policy_citations_point_to_existing_documents(self):
        document_ids = {doc["id"] for doc in kb._policy_docs()}
        for entry in kb._entries():
            with self.subTest(entry=entry["id"]):
                result = kb.answer(entry["question"])
                for citation in result["citations"]:
                    self.assertIn(citation["doc_id"], document_ids)
                    self.assertTrue(citation["excerpt"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
