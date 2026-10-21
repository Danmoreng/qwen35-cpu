// Local CPU inference; model bytes stay in this worker and are shared by pthreads.
let module, mounted = false, loading = false, running = false, cancelled = false;
const CACHE = 'qev-local-wasm-v1';
async function openModelCache() {
  // Large Cache API entries may be rejected in private browser contexts.
  // OPFS stores the checkpoint as a file and feeds WORKERFS without a JS copy.
  if (navigator.storage?.getDirectory) try {
    const root = await navigator.storage.getDirectory();
    return {kind: 'opfs', root: await root.getDirectoryHandle(CACHE, {create: true})};
  } catch (error) { send('log', {text: `OPFS unavailable: ${error.message}`}); }
  try { return {kind: 'cache-api', store: await caches.open(CACHE)}; } catch { return undefined; }
}
async function clearModelCache() {
  await caches.delete(CACHE);
  if (navigator.storage?.getDirectory) {
    const root = await navigator.storage.getDirectory();
    try { await root.removeEntry(CACHE, {recursive: true}); }
    catch (error) { if (error.name !== 'NotFoundError') throw error; }
  }
}
// MessageChannel yields without the nested setTimeout minimum delay.
const taskChannel = new MessageChannel();
let resumeTask;
taskChannel.port1.onmessage = () => { const resume = resumeTask; resumeTask = undefined; resume(); };
const nextTask = () => new Promise(resolve => { resumeTask = resolve; taskChannel.port2.postMessage(0); });
const send = (type, value) => postMessage({type, ...value});
const call = (name, types = [], args = []) => {
  const result = JSON.parse(module.ccall(name, 'string', types, args));
  if (result.error) throw new Error(result.error);
  return result;
};
async function getFile(file, cache, progress) {
  const url = `/model/${file.name}?sha256=${file.sha256}`;
  const key = `${file.sha256}-${file.name}`;
  let blob, response, cached = false;
  if (cache?.kind === 'opfs') try {
    blob = await (await cache.root.getFileHandle(key)).getFile(); cached = true;
  } catch (error) { if (error.name !== 'NotFoundError') throw error; }
  if (cache?.kind === 'cache-api') { response = await cache.store.match(url); cached = !!response; }
  if (!blob && !response) {
    response = await fetch(url);
    if (!response.ok) throw new Error(`Cannot load ${file.name}: HTTP ${response.status}`);
  }
  const cacheWrite = !cached && cache?.kind === 'cache-api' ? cache.store.put(url, response.clone()).catch(error => {
    send('log', {text: `Model cache write failed: ${error.message}`});
  }) : Promise.resolve();
  if (response?.body) {
    let bytes = 0;
    response = new Response(response.body.pipeThrough(new TransformStream({transform(chunk, controller) {
      bytes += chunk.byteLength; progress(bytes); controller.enqueue(chunk);
    }})));
  }
  if (!blob && cache?.kind === 'opfs') {
    const handle = await cache.root.getFileHandle(key, {create: true});
    try {
      await response.body.pipeTo(await handle.createWritable());
      blob = await handle.getFile();
    } catch (error) {
      await cache.root.removeEntry(key).catch(() => {}); throw error;
    }
  }
  // File/Blob-backed WORKERFS avoids a checkpoint-sized MEMFS allocation.
  if (!blob) blob = await response.blob();
  await cacheWrite;
  if (blob.size !== file.bytes) {
    if (cache?.kind === 'cache-api') await cache.store.delete(url);
    if (cache?.kind === 'opfs') await cache.root.removeEntry(key);
    throw new Error(`Unexpected size for ${file.name}; clear the local model cache.`);
  }
  // The C++ artifact reader also validates every tensor's checksum before use.
  if (file.name !== 'model.q35h') {
    const hash = await crypto.subtle.digest('SHA-256', await blob.arrayBuffer());
    const actual = [...new Uint8Array(hash)].map(x => x.toString(16).padStart(2, '0')).join('');
    if (actual !== file.sha256) throw new Error(`Checksum mismatch: ${file.name}`);
  }
  return {name: file.name, data: blob, cached};
}
onmessage = async ({data}) => {
  try {
    if (data.type === 'cancel') { cancelled = true; return; }
    if (data.type === 'clear-cache') {
      if (loading || running) throw new Error('Wait until the current operation finishes.');
      await clearModelCache(); send('cache-cleared', {}); return;
    }
    if (data.type === 'load') {
      if (loading || running || module) throw new Error('Reload the worker before loading another engine.');
      loading = true;
      const mode = data.mode;
      if (!['scalar', 'simd', 'threads', 'relaxed', 'relaxed-threads', 'fma', 'fma-threads'].includes(mode)) throw new Error('Unknown engine mode');
      if (mode.endsWith('threads') && !crossOriginIsolated) throw new Error('Threads require cross-origin isolation.');
      if (mode.startsWith('fma') && !(await import('./capabilities.js')).fusedFmaAvailable())
        throw new Error('This browser does not provide the required fused FMA semantics. Select optimized DOT4 instead.');
      importScripts(`./${mode}/qev_wasm.js`);
      module = await createQevModule({mainScriptUrlOrBlob: new URL(`./${mode}/qev_wasm.js`, location.href).href,
        locateFile: path => new URL(`./${mode}/${path}`, location.href).href,
        printErr: text => send('log', {text})});
      const response = await fetch('/model/manifest.json');
      if (!response.ok) throw new Error('Local model manifest is unavailable');
      const manifest = await response.json();
      if (manifest.format !== 'qev-local-wasm/1') throw new Error('Unsupported local manifest');
      let cache;
      if (data.cache) cache = await openModelCache();
      const blobs = []; let loaded = 0, lastProgress = 0;
      const total = manifest.files.reduce((sum, x) => sum + x.bytes, 0);
      for (const file of manifest.files) {
        send('progress', {phase: `Loading ${file.name}`, loaded, total});
        let blob;
        const progress = bytes => {
          if (performance.now() - lastProgress < 80) return;
          lastProgress = performance.now(); send('progress', {phase: `Loading ${file.name}`, loaded: loaded + bytes, total});
        };
        try { blob = await getFile(file, cache, progress); }
        catch (error) {
          // A full or blocked cache must not prevent local inference.
          if (!cache) throw error;
          send('log', {text: `Cache unavailable: ${error.message}`});
          cache = undefined; blob = await getFile(file, undefined, progress);
        }
        loaded += file.bytes; blobs.push(blob);
        send('progress', {phase: blob.cached ? 'Loaded from browser cache' : `Loaded ${file.name}`, loaded, total});
      }
      module.FS.mkdir('/model');
      module.FS.mount(module.WORKERFS, {blobs}, '/model'); mounted = true;
      send('progress', {phase: 'Validating tensors and loading the CPU engine', loaded: total, total});
      const start = performance.now();
      if (data.chunk !== undefined || data.profile)
        call('qev_options', ['number', 'number'], [data.chunk ?? 32, data.profile ? 1 : 0]);
      const result = call('qev_load', ['string', 'number', 'number'], ['/model', data.threads, data.context]);
      // All weights/configuration are owned by C++; release Blob/file references.
      module.FS.unmount('/model'); mounted = false; blobs.length = 0;
      loading = false;
      send('ready', {info: {...result, mode, load_ms: performance.now() - start,
        model_sha256: manifest.model_sha256}}); return;
    }
    if (data.type === 'prepare' || data.type === 'decide') {
      if (!module || loading) throw new Error('Load the model first');
      if (running) throw new Error('A decision is already running');
      running = true; cancelled = false;
      const started = performance.now();
      // Preserve raw JSON order, including integer-looking question/option keys.
      const body = typeof data.request === 'string' ? data.request : JSON.stringify(data.request);
      let result = call('qev_begin', ['string'], [body]);
      if (data.type === 'prepare') {
        const debug = call('qev_debug'); call('qev_cancel'); running = false;
        send('prepared', {id: data.id, debug}); return;
      }
      while (result.status === 'running') {
        // Yield between bounded prefill chunks for cancel/progress messages.
        await nextTask();
        if (cancelled) { result = call('qev_cancel'); break; }
        result = call('qev_step');
        if (result.status === 'running') send('inference-progress', result);
      }
      const debug = data.debug ? call('qev_debug') : undefined;
      running = false;
      send('result', {id: data.id, result, debug, worker_ms: performance.now() - started}); return;
    }
    throw new Error('Unknown worker operation');
  } catch (error) {
    if (running && module) { try { call('qev_cancel'); } catch {} }
    running = loading = false;
    if (mounted) { module.FS.unmount('/model'); mounted = false; }
    send('error', {message: error.message || String(error), id: data.id});
  }
};
