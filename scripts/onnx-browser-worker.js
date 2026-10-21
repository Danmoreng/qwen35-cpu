// Benchmark adapter for the external Transformers.js / ONNX Runtime engine.
// This does not add a GPU backend to the native CPU engine.
import {AutoModelForCausalLM, Tensor, env} from './runtime/transformers.js';
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = new URL('./', location.href).href;
env.useBrowserCache = false;
env.useWasmCache = false;
let model, config;
onmessage = async ({data}) => {
  try {
    if (data.type === 'load') {
      const suffix = {plain: '', jsep: '.jsep', asyncify: '.asyncify'}[data.ortBuild || 'asyncify'];
      if (suffix === undefined) throw new Error('Unknown ORT build');
      env.backends.onnx.wasm.wasmPaths = {
        mjs: new URL(`./runtime/ort-wasm-simd-threaded${suffix}.mjs`, location.href).href,
        wasm: new URL(`./runtime/ort-wasm-simd-threaded${suffix}.wasm`, location.href).href,
      };
      env.backends.onnx.wasm.numThreads = data.threads;
      config = await (await fetch('./model/qev_config.json')).json();
      const adapter = data.provider === 'webgpu' ? await navigator.gpu?.requestAdapter() : null;
      if (data.provider === 'webgpu' && (!adapter || adapter.info.isFallbackAdapter))
        throw new Error('No hardware WebGPU adapter is available');
      const started = performance.now();
      model = await AutoModelForCausalLM.from_pretrained('model', {
        device: data.provider, dtype: data.dtype, local_files_only: true,
        session_options: {graphOptimizationLevel: 'all'},
        progress_callback: progress => {
          if (['initiate', 'done', 'ready'].includes(progress.status)) postMessage({type: 'progress', progress});
        },
      });
      postMessage({type: 'ready', load_ms: performance.now() - started,
        sessions: Object.fromEntries(Object.entries(model.sessions).map(([key, session]) =>
          [key, {inputs: session.inputNames, outputs: session.outputNames}])),
        adapter: adapter ? {vendor: adapter.info.vendor, architecture: adapter.info.architecture,
          description: adapter.info.description, isFallbackAdapter: adapter.info.isFallbackAdapter} : null,
      });
    } else if (data.type === 'decide') {
      const started = performance.now();
      const ids = data.tokens;
      const input_ids = new Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]);
      const attention_mask = new Tensor('int64', new BigInt64Array(ids.length).fill(1n), [1, ids.length]);
      const num_logits_to_keep = new Tensor('int64', [1n], []);
      // Fresh zero state for every call; last prompt slot only, no generation.
      const outputs = await model({input_ids, attention_mask, num_logits_to_keep});
      const tensor = outputs.logits;
      if (tensor.location === 'gpu-buffer') await tensor.getData();
      const values = tensor.data;
      const offset = values.length - 248320;
      const names = data.names;
      const logits = names.map((_, i) => values[offset + config.letter_token_ids[i]]);
      const temperature = config.temperatures[data.questionType || 'choice'];
      const maximum = Math.max(...logits);
      const exponentials = logits.map(x => Math.exp((x - maximum) / temperature));
      const denominator = exponentials.reduce((a, b) => a + b, 0);
      const probabilities = Object.fromEntries(names.map((name, i) => [name, exponentials[i] / denominator]));
      const winner = names[logits.indexOf(maximum)];
      const worker_ms = performance.now() - started;
      const disposed = new Set();
      for (const value of Object.values(outputs)) {
        if (value?.dispose && !disposed.has(value)) { value.dispose(); disposed.add(value); }
      }
      postMessage({type: 'result', tokens: ids.length, logits, probabilities, choice: winner,
        worker_ms, output_tokens: 0, decode_forwards: 0});
    } else throw new Error('Unknown operation');
  } catch (error) {
    postMessage({type: 'error', message: error.stack || String(error)});
  }
};
