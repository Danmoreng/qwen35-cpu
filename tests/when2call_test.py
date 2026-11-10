"""Checks for the decision adapter and scoring, without model prerequisites."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("when2call", Path(__file__).parents[1] / "scripts/when2call.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
summary_spec = importlib.util.spec_from_file_location(
    "when2call_summary", Path(__file__).parents[1] / "scripts/summarize-when2call.py")
summary = importlib.util.module_from_spec(summary_spec)
summary_spec.loader.exec_module(summary)


class When2CallTests(unittest.TestCase):
    def item(self):
        return dict(uuid="frozen-task-id", source="source", source_id="sample-0",
                    question="What is the weather?", correct_answer="tool_call",
                    tools=['{"name": "weather", "parameters": {}}'],
                    answers=dict(direct="It is sunny.", tool_call='{"name": "weather", "arguments": {}}',
                                 request_for_info="Which city?", cannot_answer="I cannot check that."),
                    target_tool="HIDDEN_TOOL", orig_tools=["HIDDEN_ORIGINAL"], held_out_param="HIDDEN_PARAMETER")

    def test_no_answer_key_or_generation_metadata_enters_request(self):
        item = self.item()
        first = bench.request_for(item)
        changed = copy.deepcopy(item)
        changed["correct_answer"] = "cannot_answer"
        changed["target_tool"] = "DIFFERENT_HIDDEN_TOOL"
        self.assertEqual(first["request"], bench.request_for(changed)["request"])
        text = str(first["request"])
        self.assertNotIn("HIDDEN", text)
        self.assertNotIn("source_id", text)
        self.assertEqual(first, bench.request_for(item))

    def test_candidate_texts_and_category_mapping_survive_shuffle(self):
        item = self.item()
        task = bench.request_for(item)
        criteria = task["request"]["questions"]["decision"]["criteria"]
        self.assertEqual(len(criteria), 4)
        self.assertEqual(set(criteria.values()), set(item["answers"].values()))
        for key, category in task["option_categories"].items():
            self.assertEqual(criteria[key], item["answers"][category])

    def test_invalid_and_unsupported_direct_predictions_are_not_dropped(self):
        rows = [dict(expected="tool_call", predicted="tool_call", correct=True, valid=True,
                     latency_ms=2, no_available_tools=False, brier=.1),
                dict(expected="cannot_answer", predicted="direct", correct=False, valid=True,
                     latency_ms=4, no_available_tools=True, brier=1.5),
                dict(expected="request_for_info", predicted=None, correct=False, valid=False,
                     latency_ms=6, no_available_tools=False)]
        result = bench.metrics(rows)
        self.assertEqual(result["requests"], 3)
        self.assertEqual(result["accuracy"], 1/3)
        self.assertEqual(result["valid"], 2)
        self.assertEqual(result["per_category"]["direct"]["support"], 0)
        self.assertIsNone(result["per_category"]["direct"]["recall"])
        self.assertEqual(result["confusion"]["request_for_info"]["invalid"], 1)
        self.assertEqual(result["no_tool_hallucination_rate"], 0)
        self.assertEqual(result["median_latency_ms"], 4)

    def evidence(self, folder):
        task = bench.request_for(self.item())
        task["native_tokens"] = 20
        probs = {key: .7 if category == "tool_call" else .1
                 for key, category in task["option_categories"].items()}
        chosen = max(probs, key=probs.get)
        answer = dict(answers=dict(decision=dict(probabilities=probs, choice=chosen)),
                      usage=dict(input_tokens=20, output_tokens=0), cache=dict(read_tokens=0))
        response_text = json.dumps(answer)
        record = dict(id=task["id"], expected="tool_call", predicted="tool_call", correct=True,
                      valid=True, latency_ms=2, status=200, no_available_tools=False,
                      probs={task["option_categories"][key]: value for key, value in probs.items()},
                      input_tokens=20, api_choice="tool_call", api_choice_disagrees=False,
                      top_probability_tie=False, brier=.12,
                      response_sha256=hashlib.sha256(response_text.encode()).hexdigest())
        response = dict(id=task["id"], request=dict(task["request"], model="qev-0.8b-Q4"),
                        status=200, response_text=response_text)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); port = sock.getsockname()[1]
        bench.save(folder / "identity.json", dict(affinity=list(range(6)), selected_ids=[task["id"]],
                   precision="Q4", command=["server", "--port", str(port)]))
        bench.save(folder / "status.json", dict(status="complete"))
        bench.save(folder / "summary.json", bench.metrics([record]))
        (folder / "results.jsonl").write_text(json.dumps(record) + "\n")
        (folder / "responses.jsonl").write_text(json.dumps(response) + "\n")
        return task

    def test_raw_validation_rejects_changed_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = self.evidence(folder)
            summary.validate(folder, {task["id"]: task}, [task["id"]])
            record = json.loads((folder / "results.jsonl").read_text())
            record["predicted"] = "direct"
            (folder / "results.jsonl").write_text(json.dumps(record) + "\n")
            with self.assertRaises(AssertionError):
                summary.validate(folder, {task["id"]: task}, [task["id"]])

    def test_interrupted_prefix_requires_explicit_partial_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            task = self.evidence(folder)
            identity = json.loads((folder / "identity.json").read_text())
            identity["selected_ids"].append("unfinished-task")
            bench.save(folder / "identity.json", identity)
            bench.save(folder / "status.json", dict(status="interrupted"))
            (folder / "summary.json").unlink()
            with self.assertRaises(AssertionError):
                summary.validate(folder, {task["id"]: task}, [task["id"]])
            result, _, _ = summary.validate(folder, {task["id"]: task}, [task["id"]], partial=True)
            self.assertEqual(result["requests"], 1)


if __name__ == "__main__":
    unittest.main()
