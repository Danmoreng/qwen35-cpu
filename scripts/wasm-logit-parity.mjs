// Compare FP32 fields, rather than rounded answer probabilities.
import {isDeepStrictEqual} from 'node:util';
export const floatBits = value => new Uint32Array(new Float32Array([value]).buffer)[0];
export function baselineMismatches(actual, expected) {
  const mismatches = [];
  if (actual.length !== expected.length) return [{field: 'request_count'}];
  for (let r = 0; r < actual.length; ++r) {
    const a = actual[r], b = expected[r];
    for (const [field, x, y] of [['prompt_tokens', a.debug.prompt_tokens, b.debug.prompt_tokens],
      ['usage', a.result.usage, b.result.usage], ['answers', a.result.answers, b.result.answers]])
      if (!isDeepStrictEqual(x, y)) mismatches.push({request: r, field});
    const bits = item => item.debug.option_logit_bits || item.debug.option_logits.map(row => row.map(floatBits));
    if (!isDeepStrictEqual(bits(a), bits(b))) mismatches.push({request: r, field: 'option_logit_bits', actual: bits(a), expected: bits(b)});
  }
  return mismatches;
}
