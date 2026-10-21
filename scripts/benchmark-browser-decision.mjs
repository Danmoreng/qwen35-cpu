// Run all performance measurements through benchmark-inference-seq.ps1.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL} from 'node:url';
const args = Object.fromEntries(process.argv.slice(2).reduce((out, x, i, all) => {
  if (i % 2 === 0) out.push([x.replace(/^--/, ''), all[i + 1]]); return out;
}, []));
for (const key of ['profile-json', 'playwright-module', 'inputs-file', 'dtype', 'provider'])
  if (!args[key]) throw new Error(`Missing --${key}`);
const inputs = JSON.parse(fs.readFileSync(args['inputs-file'], 'utf8'));
const launchArgs = args['gpu-flags'] === 'true'
  ? ['--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan'] : [];
const {chromium} = await import(pathToFileURL(path.resolve(args['playwright-module'])).href);
const browser = await chromium.launch({headless: args.headless !== 'false', args: launchArgs,
  ...(args.browser ? {executablePath: args.browser} : {})});
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const resources = [], logs = [], errors = [];
const engine = args.engine || 'transformers.js';
const profile = {engine, provider: args.provider, dtype: args.dtype,
  threads: Number(args.threads || 8), browser: browser.version(), launch_args: launchArgs,
  headless: args.headless !== 'false', input_sha256: sha(args['inputs-file']), outputs: [],
  ort_build: args['ort-build'] || 'asyncify',
  flash_attn: args['flash-attn'] || 'auto',
  custom_mode: args.mode || 'relaxed-threads', chunk: Number(args.chunk || 32), profile_ops: args['profile-ops'] === 'true',
  session_warmup: args['session-warmup'] !== 'false',
  runtime_version: JSON.parse(fs.readFileSync('.cache/wasm-browser/node_modules/@huggingface/transformers/package.json')).version,
  ort_version: JSON.parse(fs.readFileSync('.cache/wasm-browser/node_modules/onnxruntime-web/package.json')).version};
try {
  const page = await browser.newPage();
  page.on('request', r => resources.push(r.url()));
  page.on('console', e => logs.push({type: e.type(), message: e.text()}));
  page.on('pageerror', e => errors.push(e.message));
  await page.goto(args.url || `http://127.0.0.1:8765/${engine === 'llama.cpp' ? 'llama-bench/' : 'onnx-bench/'}`);
  const result = await page.evaluate(async ({provider, dtype, threads, inputs, warmup, ortBuild, engine, weights, ubatch, flashAttn, mode, chunk, profileOps}) => {
    const worker = new Worker(engine === 'custom' ? '../worker.js' : './worker.js',
      engine === 'transformers.js' ? {type: 'module'} : {});
    const events = [];
    let pending;
    worker.onmessage = ({data}) => {
      if (data.type === 'error') { pending?.reject(new Error(data.message)); pending = null; }
      else if (data.type === 'ready' || data.type === 'result') { pending?.resolve(data); pending = null; }
      else events.push(data);
    };
    worker.onerror = e => { pending?.reject(new Error(e.message)); pending = null; };
    const rpc = data => new Promise((resolve, reject) => {
      const timer = setTimeout(() => { worker.terminate(); reject(new Error(`Timed out: ${data.type}`)); }, 240000);
      pending = {resolve: x => {clearTimeout(timer); resolve(x);}, reject: e => {clearTimeout(timer); reject(e);}};
      worker.postMessage(data);
    });
    try {
      const ready = await rpc({type: 'load', provider, dtype, threads, ortBuild, weights, ubatch, flashAttn,
        mode, chunk, profile: profileOps, context: 2048, cache: false});
      const decide = input => rpc(engine === 'custom'
        ? {type: 'decide', request: input.request, debug: true} : {type: 'decide', ...input});
      if (warmup) for (const input of inputs) await decide(input);
      const outputs = [];
      for (const input of inputs) outputs.push(await decide(input));
      return {ready, outputs, events};
    } finally { worker.terminate(); }
  }, {provider: args.provider, dtype: args.dtype, threads: profile.threads, inputs,
    warmup: profile.session_warmup, ortBuild: profile.ort_build, engine,
    weights: args.weights ? path.basename(args.weights) : null, ubatch: Number(args.ubatch || 512),
    flashAttn: {auto: -1, off: 0, on: 1}[profile.flash_attn],
    mode: profile.custom_mode, chunk: profile.chunk, profileOps: profile.profile_ops});
  Object.assign(profile, result);
  result.outputs.forEach((o, i) => { profile[`request_${i}_ms`] = o.worker_ms; });
  const cdp = await browser.newBrowserCDPSession();
  const {processInfo} = await cdp.send('SystemInfo.getProcessInfo');
  profile.processes = processInfo.map(p => {
    const record = {id: p.id, type: p.type};
    try { record.cpu_allowed_list = fs.readFileSync(`/proc/${p.id}/status`, 'utf8').match(/^Cpus_allowed_list:\s*(.*)$/m)?.[1]; } catch {}
    return record;
  });
  await cdp.detach();
  if (resources.some(url => new URL(url).hostname !== '127.0.0.1')) throw new Error('Unexpected external resource');
  if (errors.length) throw new Error(errors.join('\n'));
  const artifacts = engine === 'transformers.js' ? [
    'scripts/onnx-browser-worker.js',
    'dist/wasm/onnx-bench/runtime/transformers.js',
    `dist/wasm/onnx-bench/runtime/ort-wasm-simd-threaded${{plain: '', jsep: '.jsep', asyncify: '.asyncify'}[profile.ort_build]}.mjs`,
    `dist/wasm/onnx-bench/runtime/ort-wasm-simd-threaded${{plain: '', jsep: '.jsep', asyncify: '.asyncify'}[profile.ort_build]}.wasm`,
    'dist/qev-onnx/export.json',
  ] : engine === 'llama.cpp' ? ['scripts/llama-browser-worker.js',
    'dist/wasm/llama-bench/runtime/llama_browser.js', 'dist/wasm/llama-bench/runtime/llama_browser.wasm',
    'dist/wasm/llama-bench/runtime/upstream.txt'] : ['web/worker.js', 'web/capabilities.js',
    'src/cpu/wasm_fma.h', 'src/cpu/q4_dot4_wasm.cpp', 'src/cpu/delta_net_wasm.cpp',
    'src/wasm/decision.cpp', 'cmake/Wasm.cmake',
    `dist/wasm/${profile.custom_mode}/qev_wasm.js`, `dist/wasm/${profile.custom_mode}/qev_wasm.wasm`];
  artifacts.push('scripts/benchmark-browser-decision.mjs');
  if (args.weights) artifacts.push(args.weights);
  profile.artifact_sha256 = Object.fromEntries(artifacts.map(file => [file, sha(file)]));
} catch (error) {
  profile.failure = error.stack || String(error);
  process.exitCode = 1;
} finally {
  Object.assign(profile, {resources, logs, errors});
  fs.writeFileSync(args['profile-json'], JSON.stringify(profile, null, 2) + '\n');
  console.log(JSON.stringify({engine: profile.engine, dtype: profile.dtype, provider: profile.provider,
    request_ms: profile.outputs.map(x => x.worker_ms), failure: profile.failure}));
  await browser.close();
}
