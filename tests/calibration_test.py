import importlib.util
from pathlib import Path
import subprocess
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
spec = importlib.util.spec_from_file_location('fit', ROOT/'scripts/fit-dot4-quant.py')
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


class CalibrationTest(unittest.TestCase):
    def test_independent_basis(self):
        h = np.array([[1.]])
        for _ in range(7):
            h = np.block([[h, h], [h, -h]])
        r = h @ np.diag(fit.signs(128)) / np.sqrt(128)
        x = np.random.default_rng(7).normal(size=(16, 128))
        w = np.random.default_rng(9).normal(size=(8, 128))
        np.testing.assert_allclose(fit.rotate(x), x@r.T, atol=1e-13)
        np.testing.assert_allclose(x@w.T, fit.rotate(x)@fit.rotate(w).T, atol=1e-12)
        g = x.T@x/len(x)
        np.testing.assert_allclose(np.mean(fit.rotate(x)**2, axis=0), np.diag(r@g@r.T), atol=1e-13)
        self.assertGreater(np.max(np.abs(np.diag(g)-np.diag(r@g@r.T))), .1)

    def test_native_basis(self):
        native = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT/'build/q4_quantizer_test.exe'
        output = subprocess.check_output([str(native), '--basis-fixture'], text=True)
        x = (np.arange(128)-64)*.03125
        np.testing.assert_allclose(np.fromstring(output, sep=' '), fit.rotate(x), atol=1e-6)


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
