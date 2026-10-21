// CPU-only benchmark adapter around an unmodified llama.cpp WASM build.
let module, config;
onmessage = async ({data}) => {
  try {
    if (data.type === 'load') {
      importScripts('./runtime/llama_browser.js');
      module = await createLlamaModule({
        mainScriptUrlOrBlob: new URL('./runtime/llama_browser.js', location.href).href,
        locateFile: path => new URL(`./runtime/${path}`, location.href).href,
        printErr: text => postMessage({type: 'log', text}),
      });
      config = await (await fetch('./qev_config.json')).json();
      const response = await fetch(`./models/${data.weights}`);
      if (!response.ok) throw new Error('GGUF download failed');
      let blob = await response.blob();
      module.FS.mkdir('/model');
      module.FS.mount(module.WORKERFS, {blobs: [{name: 'model.gguf', data: blob}]}, '/model');
      const info = JSON.parse(module.ccall('bench_load', 'string', ['string', 'number', 'number', 'number'],
        ['/model/model.gguf', data.threads, data.ubatch || 512, data.flashAttn ?? -1]));
      if (info.error) throw new Error(info.error);
      module.FS.unmount('/model'); blob = null;
      postMessage({type: 'ready', ...info});
    } else if (data.type === 'decide') {
      const started = performance.now();
      const names = data.names;
      const result = JSON.parse(module.ccall('bench_score', 'string', ['string'],
        [JSON.stringify({tokens: data.tokens, letter_ids: config.letter_token_ids.slice(0, names.length)})]));
      if (result.error) throw new Error(result.error);
      const maximum = Math.max(...result.logits);
      const temperature = config.temperatures[data.questionType || 'choice'];
      const exponentials = result.logits.map(x => Math.exp((x - maximum) / temperature));
      const denominator = exponentials.reduce((a, b) => a + b, 0);
      postMessage({type: 'result', ...result, tokens: data.tokens.length,
        probabilities: Object.fromEntries(names.map((n, i) => [n, exponentials[i] / denominator])),
        choice: names[result.logits.indexOf(maximum)], worker_ms: performance.now() - started});
    } else throw new Error('Unknown operation');
  } catch (error) { postMessage({type: 'error', message: error.stack || String(error)}); }
};
