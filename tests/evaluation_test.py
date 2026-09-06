"""Synthetic regression tests; no model or network required."""
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from evaluation_common import select_windows, source_identity, window_folder
spec = importlib.util.spec_from_file_location('compare', ROOT / 'scripts/compare-logit-dumps.py')
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


class EvaluationTest(unittest.TestCase):
    def test_selection(self):
        windows = [dict(window=i) for i in range(16)]
        self.assertEqual(select_windows(windows, 8, 4), windows[8:12])
        for start, count in [(-1, 2), (0, 0), (16, 1), (14, 3)]:
            with self.assertRaises(ValueError):
                select_windows(windows, start, count)

    def test_archive_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            identity = source_identity(tmp)
            self.assertIsNone(identity['revision'])
            self.assertEqual(len(identity['source_tree_sha256']), 64)

    def test_portable_window_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'windows'/'zero'
            folder.mkdir(parents=True)
            resolved = window_folder(dict(folder='windows\\zero'), Path(tmp)/'manifest.json')
            self.assertEqual(resolved, folder.resolve())

    def compare_dumps(self, a, b, target_b=0, truncate=False):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name, values, target in [('a', a, 0), ('b', b, target_b)]:
                data = compare.HEADER.pack(compare.MAGIC, 1, len(values), 1)
                data += compare.TARGET.pack(target) + np.array(values, dtype='<f4').tobytes()
                (folder/name).write_bytes(data[:-1] if truncate and name == 'b' else data)
            result = subprocess.run([sys.executable, str(ROOT/'scripts/compare-logit-dumps.py'),
                                     '--teacher', str(folder/'a'), '--candidate', str(folder/'b')],
                                    capture_output=True, text=True)
            return result.returncode, json.loads(result.stdout) if result.returncode == 0 else None

    def test_identical_and_shift(self):
        for shift in [0, 2]:
            code, result = self.compare_dumps([0, 1, 2], [shift, shift+1, shift+2])
            self.assertEqual(code, 0)
            self.assertAlmostEqual(result['kld']['mean'], 0)
            self.assertEqual(result['agreement']['top5_overlap'], 1)

    def test_stable_ties(self):
        np.testing.assert_array_equal(compare.top_indices(np.zeros(20), 5), np.arange(5))

    def test_hand_kl(self):
        code, result = self.compare_dumps([0, 0], [0, math.log(3)])
        self.assertEqual(code, 0)
        self.assertAlmostEqual(result['kld']['mean'], .5*math.log(4/3), places=6)

    def test_invalid(self):
        for b, target, truncated in [([1, 2], 1, False), ([1, 2], 0, True),
                                     ([1, float('nan')], 0, False),
                                     ([1, float('inf')], 0, False), ([1, 2, 3], 0, False)]:
            self.assertNotEqual(self.compare_dumps([1, 2], b, target, truncated)[0], 0)

    def test_unequal_window_aggregation(self):
        spec = importlib.util.spec_from_file_location('report', ROOT/'scripts/quality-report.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            windows = [dict(window=0, scored_tokens=1), dict(window=1, scored_tokens=3)]
            (root/'commands.json').write_text(json.dumps(dict(windows=windows,
                checkpoint_sha256='a', reference_checkpoint_sha256='a')))
            for w in windows:
                folder = root/str(w['window'])
                folder.mkdir()
                # One token at NLL=1 and three at NLL=3: aggregate NLL=2.5.
                nll = 1 if w['window'] == 0 else 3
                data = 'position,target_token,target_nll_candidate,kld_teacher_candidate,top1_equal\n'
                data += ''.join(f'{i},0,{nll},0.1,True\n' for i in range(w['scored_tokens']))
                for name in ['BF16-vs-native.csv', 'BF16-vs-llama.csv', 'llama-test-vs-native.csv']:
                    (folder/name).write_text(data)
            result = module.report(root)
            self.assertEqual(result['positions'], 4)
            self.assertAlmostEqual(result['candidate_ppl'], math.exp(2.5))
            self.assertTrue(result['strict_pass'])


if __name__ == '__main__':
    unittest.main()
