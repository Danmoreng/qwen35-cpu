import assert from 'node:assert/strict';
import {baselineMismatches, floatBits} from '../scripts/wasm-logit-parity.mjs';
const record = {debug: {prompt_tokens: [[1, 2]], option_logits: [[1]], option_logit_bits: [[floatBits(1)]]},
  result: {usage: {input_tokens: 2}, answers: {q: {choice: 'left'}}}};
const baseline = [record], copy = () => structuredClone(baseline);
assert.deepEqual(baselineMismatches(copy(), baseline), []);
let candidate = copy(); candidate[0].debug.option_logit_bits[0][0]++;
assert.equal(baselineMismatches(candidate, baseline)[0].field, 'option_logit_bits');
candidate = copy(); candidate[0].debug.prompt_tokens[0][0]++;
assert.equal(baselineMismatches(candidate, baseline)[0].field, 'prompt_tokens');
candidate = copy(); candidate[0].result.answers.q.choice = 'right';
assert.equal(baselineMismatches(candidate, baseline)[0].field, 'answers');
candidate = copy(); delete candidate[0].debug.option_logit_bits;
assert.deepEqual(baselineMismatches(candidate, baseline), []);
assert.notEqual(floatBits(-0), floatBits(0));
assert.equal(baselineMismatches([], baseline)[0].field, 'request_count');
console.log('FP32 baseline gate detects one-bit changes, protocol changes and legacy profiles');
