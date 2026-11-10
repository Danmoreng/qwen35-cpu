"""Model-free checks for external cohort selection and metric interpretation."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

scripts = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("comparison", scripts / "compare-when2call.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


class ComparisonTests(unittest.TestCase):
    def variants(self):
        return [dict(uuid=str(i), source_id="shared-source-id", correct_answer=category)
                for i, category in enumerate(("tool_call", "request_for_info", "cannot_answer"))]

    def test_aa_filter_preserves_sibling_variants(self):
        kept, removed = comparison.select_cohort(self.variants(), {"shared-source-id"})
        self.assertEqual([row["correct_answer"] for row in removed], ["request_for_info"])
        self.assertEqual({row["correct_answer"] for row in kept}, {"tool_call", "cannot_answer"})

    def test_missing_or_duplicated_ask_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            comparison.select_cohort(self.variants(), {"not-present"})
        with self.assertRaises(ValueError):
            comparison.select_cohort(self.variants()+[self.variants()[1]], {"shared-source-id"})

    def test_aa_snapshot_retains_accuracy_separate_from_index_and_decision_score(self):
        model = dict(slug="test", kind="decisionModel", name="Test", indexScore=99,
                     scores=dict(when2callAccuracy=70, when2callDecisionScore=80))
        flight = "1:" + json.dumps(dict(models=[model])) + "\n"
        html = "<script>self.__next_f.push(" + json.dumps([1, flight]) + ")</script>"
        parsed = comparison.aa_models(html)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["scores"]["when2callAccuracy"], 70)
        self.assertNotEqual(parsed[0]["scores"]["when2callAccuracy"], parsed[0]["indexScore"])

    def test_comparison_scores_explicit_choice_and_retains_rounded_argmax(self):
        rows = [dict(expected="tool_call", Q4_api_choice="cannot_answer", Q4_prediction="tool_call")]
        result = comparison.decision_counts(rows, "Q4")
        self.assertEqual(result["correct"], 0)
        self.assertEqual(result["saved_probability_argmax_correct"], 1)


if __name__ == "__main__":
    unittest.main()
