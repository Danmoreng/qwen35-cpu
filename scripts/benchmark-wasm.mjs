// Run through benchmark-inference-seq.ps1 for sequential, reproducible measurements.
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import {baselineMismatches} from './wasm-logit-parity.mjs';
const args = Object.fromEntries(process.argv.slice(2).reduce((out, x, i, all) => {
  if (i % 2 === 0) out.push([x.replace(/^--/, ''), all[i + 1]]); return out;
}, []));
for (const key of ['mode', 'requests-file', 'profile-json', 'playwright-module'])
  if (!args[key]) throw new Error(`Missing --${key}`);
const {chromium} = await import(pathToFileURL(path.resolve(args['playwright-module'])).href);
const requests = fs.readFileSync(args['requests-file'], 'utf8').trim().split('\n');
requests.forEach(JSON.parse);
const protocolInputs = args['protocol-file'] ? fs.readFileSync(args['protocol-file'], 'utf8').trim().split('\n') : [];
const protocolOutputs = args['protocol-reference'] ? fs.readFileSync(args['protocol-reference'], 'utf8').trim().split('\n').map(JSON.parse) : [];
assert.equal(protocolInputs.length, protocolOutputs.length);
const references = args['references-file'] ? fs.readFileSync(args['references-file'], 'utf8').trim().split('\n').map(JSON.parse) : null;
if (references) assert.equal(references.length, requests.length);
const baseline = args['baseline-profile'] ? JSON.parse(fs.readFileSync(args['baseline-profile'], 'utf8')) : null;
const browser = await chromium.launch({headless: true, ...(args.browser ? {executablePath: args.browser} : {})});
const logs = [], errors = [];
try {
  const page = await browser.newPage();
  page.on('console', event => { if (event.type() === 'error') errors.push(event.text()); });
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(args.url || 'http://127.0.0.1:8765');
  const collected = await page.evaluate(async ({mode, threads, context, chunk, requests, checks, protocolInputs, protocolOutputs, sessionWarmup, tuning, runtimePath}) => {
    const worker = new Worker('./worker.js');
    const events = [];
    let pending;
    let progress;
    worker.onmessage = ({data}) => {
      if (data.type === 'error') { pending?.reject(new Error(data.message)); pending = null; }
      else if (data.type === 'ready' || data.type === 'result' || data.type === 'prepared') { pending?.resolve(data); pending = null; }
      else if (data.type === 'log') events.push(data.text);
      else if (data.type === 'inference-progress') progress = data;
    };
    worker.onerror = event => { pending?.reject(new Error(event.message)); pending = null; };
    const rpc = data => new Promise((resolve, reject) => {
      progress = undefined;
      const timer = setTimeout(() => { worker.terminate(); reject(new Error(`Worker operation exceeded 180s: ${mode}/${data.type}/${data.id ?? 'load or warmup'}, progress=${JSON.stringify(progress)}`)); }, 180000);
      pending = {resolve: x => { clearTimeout(timer); resolve(x); }, reject: e => { clearTimeout(timer); reject(e); }};
      worker.postMessage(data);
    });
    try {
      const loaded = await rpc({type: 'load', mode, threads, context, chunk, tuning, runtimePath, cache: false});
      const outputs = [];
      // One untimed in-session warmup, separate from the runner's process warmup.
      if (sessionWarmup) await rpc({type: 'decide', id: 'warmup', request: requests[0]});
      const begin = performance.now();
      for (let i = 0; i < requests.length; ++i)
        outputs.push(await rpc({type: 'decide', id: i, request: requests[i], debug: true}));
      const wall = performance.now() - begin;
      const checksPassed = [];
      if (checks) {
        const base = JSON.parse(requests[0]);
        for (const invalid of [{}, {...base, encoding: 'bad'}, {...base, state: 'word '.repeat(10000)}]) {
          let rejected = false;
          try { await rpc({type: 'decide', request: invalid}); } catch { rejected = true; }
          if (!rejected) throw new Error('Invalid request was accepted');
        }
        checksPassed.push('malformed request', 'invalid encoding', 'context overflow');
        for (const tuning of [{kv_cache_bits: 8}, {kv_cache_bits: -16},
                              {kv_cache_bits: true}, {unknown_option: 0}]) {
          let rejected = false;
          try { await rpc({type: 'tune', tuning}); } catch { rejected = true; }
          if (!rejected) throw new Error('Invalid tuning was accepted');
        }
        checksPassed.push('invalid KV widths/types and unknown tuning rejected');
        const cancelled = rpc({type: 'decide', request: requests.at(-1)});
        worker.postMessage({type: 'cancel'});
        if ((await cancelled).result.status !== 'cancelled') throw new Error('Cancel did not stop inference');
        checksPassed.push('cancel');
        const recovered = await rpc({type: 'decide', request: requests[0]});
        if (JSON.stringify(recovered.result.answers) !== JSON.stringify(outputs[0].result.answers)) throw new Error('State leaked after cancellation');
        checksPassed.push('fresh state after cancel');
        for (let i = 0; i < protocolInputs.length; ++i) {
          const prepared = await rpc({type: 'prepare', request: protocolInputs[i]});
          if (JSON.stringify(prepared.debug.prompt_tokens) !== JSON.stringify(protocolOutputs[i].rows))
            throw new Error(`Protocol token mismatch: ${i}`);
        }
        if (protocolInputs.length) checksPassed.push(`${protocolInputs.length} raw JSON protocol fixtures`);
      }
      return {loaded, outputs, wall_ms: wall, checks: checksPassed, events};
    } finally { worker.terminate(); }
  }, {mode: args.mode, threads: Number(args.threads || 1), context: Number(args.context || 2048), chunk: Number(args.chunk || 32), requests, checks: args.checks === 'true', protocolInputs, protocolOutputs, sessionWarmup: args['session-warmup'] !== 'false', tuning: args.tuning ? JSON.parse(args.tuning) : undefined, runtimePath: args['runtime-path']});
  let maxLogit = 0, maxProbability = 0, questions = 0;
  const choiceChanges = [];
  const probabilities = answer => answer.type === 'noul' ? {true: answer.noul} : answer.probabilities;
  for (let i = 0; i < collected.outputs.length; ++i) {
    const actual = collected.outputs[i];
    assert.equal(actual.result.status, 'complete');
    assert.equal(actual.result.metrics.decode_forwards, 0);
    questions += Object.keys(actual.result.answers).length;
    if (!references) continue;
    const expected = references[i];
    assert.deepEqual(actual.debug.prompt_tokens, expected.debug.prompt_tokens, `Tokenizer mismatch in request ${i}`);
    assert.deepEqual(actual.result.usage, expected.result.usage);
    for (let q = 0; q < actual.debug.option_logits.length; ++q)
      actual.debug.option_logits[q].forEach((value, j) => { maxLogit = Math.max(maxLogit, Math.abs(value - expected.debug.option_logits[q][j])); });
    for (const [id, answer] of Object.entries(actual.result.answers)) {
      const reference = expected.result.answers[id];
      assert.equal(answer.type, reference.type);
      if (answer.type === 'choice' && answer.choice !== reference.choice) {
        const sorted = Object.values(reference.probabilities).sort((a,b) => b-a);
        choiceChanges.push({request: i, question: id, expected: reference.choice,
          actual: answer.choice, reference_margin: sorted[0] - sorted[1]});
      }
      for (const [key, probability] of Object.entries(probabilities(answer)))
        maxProbability = Math.max(maxProbability, Math.abs(probability - probabilities(reference)[key]));
    }
  }
  const times = collected.outputs.map(x => x.worker_ms);
  const totalTokens = collected.outputs.reduce((n, x) => n + x.result.metrics.evaluated_tokens, 0);
  const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  const profile = {engine: `wasm-${args.mode}`, threads: Number(args.threads || 1), chunk: Number(args.chunk || 32), requests: requests.length, questions,
    decisions_per_second: questions / (collected.wall_ms / 1000), mean_request_ms: times.reduce((a, b) => a + b) / times.length,
    total_wall_ms: collected.wall_ms, prefill_tokens_per_second: totalTokens / (collected.wall_ms / 1000),
    wasm_heap_bytes: Math.max(...collected.outputs.map(x => x.result.metrics.wasm_heap_bytes)),
    tuning: args.tuning ? JSON.parse(args.tuning) : {},
    reference_checked: !!references,
    max_reference_logit_delta: references ? maxLogit : null, max_reference_probability_delta: references ? maxProbability : null,
    choice_changes: choiceChanges,
    baseline_bit_mismatches: baseline ? baselineMismatches(collected.outputs, baseline.outputs) : null,
    baseline_profile_sha256: baseline ? sha(args['baseline-profile']) : null,
    browser: browser.version(), request_sha256: sha(args['requests-file']),
    session_warmup: args['session-warmup'] !== 'false',
    reference_sha256: args['references-file'] ? sha(args['references-file']) : null,
    model_sha256: collected.loaded.info.model_sha256,
    checks: collected.checks, outputs: collected.outputs, ready: collected.loaded.info, logs: [...logs, ...collected.events], errors};
  collected.outputs.forEach((value, i) => { profile[`request_${i}_ms`] = value.worker_ms; });
  try {
    const cdp = await browser.newBrowserCDPSession();
    const {processInfo} = await cdp.send('SystemInfo.getProcessInfo');
    profile.processes = processInfo.map(process => {
      const record = {id: process.id, type: process.type};
      try {
        const status = fs.readFileSync(`/proc/${process.id}/status`, 'utf8');
        record.cpu_allowed_list = status.match(/^Cpus_allowed_list:\s*(.*)$/m)?.[1];
        record.peak_rss_kib = Number(status.match(/^VmHWM:\s*(\d+)/m)?.[1]) || null;
      } catch { /* Non-Linux browsers still report the inference evidence. */ }
      return record;
    });
    await cdp.detach();
  } catch { /* Process introspection is diagnostic, not required for execution. */ }
  profile.artifacts = Object.fromEntries([
    `dist/wasm/${args['runtime-path'] || args.mode}/qev_wasm.js`, `dist/wasm/${args['runtime-path'] || args.mode}/qev_wasm.wasm`,
    'web/worker.js', 'web/app.js', 'web/capabilities.js', 'src/wasm/decision.cpp',
    'src/cpu/q4_dot4_wasm.cpp', 'src/cpu/delta_net_wasm.cpp', 'cmake/Wasm.cmake',
    'src/cpu/q8_packed_wasm.cpp', 'src/cpu/q4_h128.cpp', 'src/runtime/weights.inl',
    'src/runtime/prefill.inl', 'src/runtime/cpu_engine.inl', 'include/qwen35x/runtime/cpu_engine.h', 'src/runtime/attention_partition.h',
    'src/cpu/q4_h128_wasm.cpp', 'src/cpu/wasm_fma.h', 'src/cpu/wasm_half.h'
  ].map(file => [file, sha(file)]));
  assert.deepEqual(errors, []);
  fs.writeFileSync(args['profile-json'], JSON.stringify(profile, null, 2) + '\n');
  console.log(JSON.stringify({...profile, outputs: undefined, logs: undefined, ready: undefined}));
  // Compare probabilities: a common offset in raw logits cancels in softmax.
  // Persist diagnostic evidence before reporting a failed numerical gate.
  if (references) assert.ok(maxProbability <= .005, `Probability drift ${maxProbability}`);
  if (baseline) assert.deepEqual(profile.baseline_bit_mismatches, [], 'WASM baseline changed');
  assert.deepEqual(choiceChanges, [], 'A selected option changed');
} finally { await browser.close(); }
