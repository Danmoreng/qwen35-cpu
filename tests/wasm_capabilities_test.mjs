// Verify that the fast-FMA runtime guard rejects an unfused implementation.
import assert from 'node:assert/strict';
const actual = await import('../web/capabilities.js');
const available = actual.fusedFmaAvailable();
assert.equal(typeof available, 'boolean');
const originalInstance = WebAssembly.Instance;
try {
  WebAssembly.Instance = class {
    constructor() {
      this.exports = {fma: (a, b, c) => Math.fround(Math.fround(a * b) + c)};
    }
  };
  const unfused = await import('../web/capabilities.js?unfused-control');
  assert.equal(unfused.fusedFmaAvailable(), false);
  WebAssembly.Instance = class { constructor() { throw new Error('Unsupported SIMD'); } };
  const unsupported = await import('../web/capabilities.js?unsupported-control');
  assert.equal(unsupported.fusedFmaAvailable(), false);
} finally { WebAssembly.Instance = originalInstance; }
console.log(JSON.stringify({fused_fma_available: available, unfused_rejected: true, unsupported_rejected: true}));
