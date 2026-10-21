import {simdAvailable, relaxedDotAvailable} from './capabilities.js';
const $ = id => document.getElementById(id);
export const examples = {
  snake: {state: {game: 'snake', up: 'The snake hits the wall and dies.', left: 'The snake reaches the food safely.', right: 'The snake enters a dead end and will be trapped.'}, questions: {move: {type: 'choice', instructions: 'Which move should the snake choose?', criteria: {up: null, left: null, right: null}}}, encoding: 'per_question'},
  ticket: {state: {ticket: 'My card was charged twice for the same order. Please refund the duplicate charge.'}, questions: {dept: {type: 'choice', instructions: 'Which team should handle this ticket?', criteria: {account: 'login and access', billing: 'payments and refunds', sales: 'pricing and upgrades', tech: 'bugs and broken features'}}}, encoding: 'per_question'},
  logic: {state: 'Every red block is heavy. Block A is red. Block B is blue. No other facts are known.', questions: {heavy: {type: 'noul', instructions: 'Is block A heavy?'}, evidence: {type: 'score', instructions: 'How strongly do the facts support that block B is heavy?', criteria: ['No support', 'Weak support', 'Moderate support', 'Strong support']}}, encoding: 'per_question'},
  grid: {state: {grid: {width: 5, height: 5, coordinates: 'x increases right; y increases down'}, player: [2, 2], goal: [1, 2], walls: [[2, 1], [3, 2]]}, questions: {move: {type: 'choice', instructions: 'Which single move reaches the goal without entering a wall?', criteria: {up: 'y minus one', left: 'x minus one', right: 'x plus one', down: 'y plus one'}}}, encoding: 'per_question'}
};
let worker, ready = false, busy = false, info;
const showError = text => { $('error').textContent = text; };
function controls() {
  $('load').disabled = busy || ready; $('run').disabled = !ready || busy;
  $('cancel').disabled = !busy || !ready;
  for (const id of ['mode', 'threads', 'context']) $(id).disabled = ready || busy;
  $('clear').disabled = busy;
  $('reset').disabled = !worker;
  $('threads').disabled = ready || busy || !$('mode').value.endsWith('threads');
}
function newWorker() {
  if (worker) worker.terminate();
  worker = new Worker('./worker.js');
  worker.onerror = event => { busy = ready = false; showError(event.message); controls(); };
  worker.onmessage = ({data}) => {
    if (data.type === 'progress') { $('progress').value = data.loaded / data.total; $('loading').textContent = data.phase; }
    if (data.type === 'ready') {
      info = data.info; ready = true; busy = false;
      $('status').textContent = 'Ready'; $('progress').value = 1;
      $('loading').textContent = `Model loaded locally · ${info.threads} thread(s) · ${Math.round(info.wasm_heap_bytes / 1048576)} MiB WASM heap`;
      $('engine-info').textContent = `${info.format} · ${info.mode} · ${info.backend}\nSHA-256 ${info.model_sha256}`;
      controls();
    }
    if (data.type === 'inference-progress') $('status').textContent = `Computing question ${(data.question || 0) + 1}…`;
    if (data.type === 'result') {
      busy = false; $('status').textContent = 'Ready'; controls();
      if (data.result.status === 'complete') render(data.result, data.worker_ms);
      else $('status').textContent = 'Cancelled';
    }
    if (data.type === 'error') { busy = false; showError(data.message); controls(); }
    if (data.type === 'cache-cleared') $('loading').textContent = 'Browser model cache cleared.';
    if (data.type === 'log') console.info(data.text);
  };
}
function render(result, workerMs) {
  $('answers').replaceChildren();
  for (const [id, answer] of Object.entries(result.answers)) {
    const section = document.createElement('div'); section.className = 'question';
    const title = document.createElement('h3'); title.textContent = answer.choice ? `${id} → ${answer.choice}` : answer.type === 'noul' ? `${id} · ${(answer.noul * 100).toFixed(2)}% yes` : `${id} · score ${answer.score}`;
    section.append(title);
    const probabilities = answer.probabilities || {no: 1 - answer.noul, yes: answer.noul};
    for (const [name, p] of Object.entries(probabilities)) {
      const row = document.createElement('div'); row.className = `prob ${name === answer.choice ? 'winner' : ''}`;
      const label = document.createElement('div'); label.className = 'prob-label';
      const key = document.createElement('span'); key.textContent = name;
      const value = document.createElement('span'); value.textContent = `${(p * 100).toFixed(2)}%`;
      label.append(key, value);
      const bar = document.createElement('div'); bar.className = 'bar';
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${Math.max(0, Math.min(100, p * 100))}%`;
      bar.append(fill); row.append(label, bar); section.append(row);
    }
    if (answer.confidence !== undefined) {
      const meta = document.createElement('p'); meta.className = 'answer-meta';
      meta.textContent = `Confidence ${(answer.confidence * 100).toFixed(1)}% · Certainty ${(answer.certainty * 100).toFixed(1)}%`; section.append(meta);
    }
    $('answers').append(section);
  }
  $('metrics').replaceChildren();
  const grid = document.createElement('div'); grid.className = 'metric-grid';
  for (const [value, label] of [[`${Math.round(workerMs)} ms`, 'Decision latency'], [result.metrics.evaluated_tokens, 'Evaluated input tokens'], [`${Math.round(result.metrics.wasm_heap_bytes / 1048576)} MiB`, 'WASM heap · not total RAM']]) {
    const cell = document.createElement('div'), strong = document.createElement('strong'), caption = document.createElement('span');
    strong.textContent = value; caption.textContent = label; cell.append(strong, caption); grid.append(cell);
  }
  $('metrics').append(grid); $('raw').textContent = JSON.stringify(result, null, 2);
}
document.querySelectorAll('[data-example]').forEach(button => button.onclick = () => { $('request').value = JSON.stringify(examples[button.dataset.example], null, 2); });
$('request').value = JSON.stringify(examples.ticket, null, 2);
for (const option of $('mode').options) {
  option.disabled = (option.value.endsWith('threads') && !crossOriginIsolated) ||
    (option.value !== 'scalar' && !simdAvailable()) ||
    (option.value.startsWith('relaxed') && !relaxedDotAvailable());
}
$('mode').value = [...$('mode').options].find(x => !x.disabled).value;
const cores = navigator.hardwareConcurrency || 1;
$('threads').value = String(cores >= 16 ? 8 : cores >= 8 ? 4 : cores >= 4 ? 2 : 1);
$('load').onclick = () => {
  showError(''); ready = false; busy = true; newWorker(); controls(); $('status').textContent = 'Loading';
  const mode = $('mode').value;
  worker.postMessage({type: 'load', mode, threads: mode.endsWith('threads') ? Number($('threads').value) : 1, context: Number($('context').value), cache: $('cache').checked});
};
$('run').onclick = () => {
  try {
    const request = $('request').value; JSON.parse(request); showError(''); busy = true; controls();
    $('status').textContent = 'Computing…'; worker.postMessage({type: 'decide', request});
  } catch (error) { showError(error.message); }
};
$('cancel').onclick = () => worker.postMessage({type: 'cancel'});
$('reset').onclick = () => {
  worker?.terminate(); worker = undefined; ready = busy = false; info = undefined;
  $('progress').value = 0; $('status').textContent = 'Not loaded'; $('engine-info').textContent = '';
  $('loading').textContent = 'Engine unloaded. Select a runtime and load again.'; controls();
};
$('mode').onchange = controls;
$('clear').onclick = async () => {
  try {
    await caches.delete('qev-local-wasm-v1');
    if (navigator.storage?.getDirectory) {
      const root = await navigator.storage.getDirectory();
      try { await root.removeEntry('qev-local-wasm-v1', {recursive: true}); }
      catch (error) { if (error.name !== 'NotFoundError') throw error; }
    }
    $('loading').textContent = 'Browser model cache cleared.';
  }
  catch (error) { showError(error.message); }
};
controls();
