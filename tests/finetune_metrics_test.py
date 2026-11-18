"""Small hand-computed task scores used by the HF/native evaluation."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import unittest

module = spec_from_file_location("quality", Path(__file__).resolve().parents[1] / "scripts/evaluate-finetune.py")
quality = module_from_spec(module)
module.loader.exec_module(quality)


class Metrics(unittest.TestCase):
    def test_asr_insert_delete_substitute(self):
        panel = dict(task="asr", cases=[{"expected": "it is ready"}, {"expected": "a small bird"}])
        actual = [{"native": "it was quite ready"}, {"native": "a bird"}]
        self.assertEqual(quality.metrics(panel, actual, "native"),
                         dict(wer=.5, insertions=1, substitutions=1, deletions=1))

    def test_extraction_missing_and_invalid(self):
        panel = dict(task="extraction", cases=[{"expected": {"city": "Lyon", "fee": None}},
                                               {"expected": {"date": "Tuesday"}}])
        actual = [{"native": '{"city":"Lyon"}'}, {"native": "Tuesday"}]
        self.assertEqual(quality.metrics(panel, actual, "native"),
                         dict(field_accuracy=1/3, valid_json_object=.5))

    def test_rank_and_classification_are_independent(self):
        panel = dict(task="classification", labels=["no", "yes"], positive="yes",
                     cases=[{"expected": "yes", "group": "g"}, {"expected": "no", "group": "g"}])
        actual = [{"native": {"yes": .3, "no": .7}}, {"native": {"yes": .2, "no": .8}}]
        self.assertEqual(quality.metrics(panel, actual, "native"), dict(mrr=1., accuracy=.5, macro_f1=1/3))


if __name__ == "__main__":
    unittest.main()
