import {simdAvailable, relaxedDotAvailable, fusedFmaAvailable} from './capabilities.js';
import {catalog, samples, directions, arrows, createScenario, useSample, requestFor,
  initialCustomRequest, snakeDirections, editSnake, stepSnake, step2048} from './scenarios.js';
const $ = id => document.getElementById(id);
let scenario = createScenario('snake'), revision = 0, worker, ready = false, busy = false;
let pending, nextId = 0, autoRunning = false, autoTimer, remaining = 0, editMode = 'food';
let customDraft = JSON.stringify(initialCustomRequest, null, 2), lastResult;
const isGame = () => catalog.find(d => d.id === scenario.id).kind === 'game';
const showError = text => { $('error').textContent = text; };
const text = (tag, value, className) => {
  const node = document.createElement(tag); node.textContent = value;
  if (className) node.className = className; return node;
};
function controls() {
  $('load').disabled = busy || ready; $('run').disabled = !ready || busy || autoRunning;
  $('cancel').disabled = !pending; $('reset').disabled = !worker;
  $('clear').disabled = busy || autoRunning;
  for (const id of ['mode', 'threads', 'context']) $(id).disabled = ready || busy;
  $('threads').disabled = ready || busy || !$('mode').value.endsWith('threads');
  $('demo-run').disabled = !ready || busy || autoRunning || scenario.over;
  $('autoplay').hidden = !isGame();
  $('autoplay').disabled = !ready || scenario.over || (busy && !autoRunning);
  $('autoplay').textContent = autoRunning ? 'Pause' : 'Autoplay';
  $('autoplay').classList.toggle('running', autoRunning);
  $('new-scene').disabled = busy || autoRunning;
  for (const node of $('scene').querySelectorAll('input, textarea, select, button')) {
    node.disabled = busy || autoRunning || (isGame() && scenario.over);
    if (node.dataset.move && scenario.id === 'snake' && !snakeDirections(scenario).includes(node.dataset.move)) node.disabled = true;
  }
  $('request').disabled = busy || autoRunning;
  for (const card of $('gallery').querySelectorAll('button')) card.disabled = Boolean(pending) || autoRunning;
}
function clearResult(message = 'Change the state, then let the model decide.') {
  lastResult = undefined;
  $('answers').replaceChildren(text('p', 'No prediction for this state yet.', 'empty'));
  $('metrics').replaceChildren(); $('raw').textContent = '{}';
  $('evaluated-request').textContent = '{}'; $('result-context').textContent = message;
  updateChips();
}
function syncRequest() {
  $('request').value = scenario.id === 'custom' ? customDraft : JSON.stringify(requestFor(scenario), null, 2);
}
function stopAuto(cancel = false) {
  autoRunning = false; clearTimeout(autoTimer);
  if (cancel && pending) { pending.discard = true; worker.postMessage({type: 'cancel'}); $('status').textContent = 'Cancelling…'; }
  controls();
}
function newWorker() {
  worker?.terminate();
  const owner = new Worker('./worker.js'); worker = owner;
  owner.onerror = event => {
    if (worker !== owner) return;
    pending?.resolve(false); pending = undefined; busy = ready = false; stopAuto();
    $('status').textContent = 'Engine error'; showError(event.message); controls();
  };
  owner.onmessage = ({data}) => {
    if (worker !== owner) return;
    if (data.type === 'progress') { $('progress').value = data.loaded / data.total; $('loading').textContent = data.phase; }
    if (data.type === 'ready') {
      ready = true; busy = false; const info = data.info;
      $('status').textContent = 'Ready'; $('progress').value = 1;
      $('loading').textContent = `Loaded locally · ${info.threads} CPU thread(s) · ready for every demo`;
      $('engine-info').textContent = `${info.format} · ${info.mode} · ${info.backend}\nModel SHA-256 ${info.model_sha256}`;
      controls();
    }
    if (data.type === 'inference-progress') $('status').textContent = `Thinking · question ${(data.question || 0) + 1}…`;
    if (data.type === 'result') {
      const job = pending;
      if (!job || job.id !== data.id) return;
      pending = undefined; busy = false;
      let applied = false;
      if (data.result.status === 'complete' && !job.discard && job.revision === revision) {
        renderResult(data.result, data.worker_ms, job);
        if (job.apply) {
          const move = data.result.answers.move?.choice;
          if (!Object.hasOwn(job.request.questions.move.criteria, move)) showError('The model did not return an available move.');
          else {
            const before = scenario.turns;
            scenario = scenario.id === 'snake' ? stepSnake(scenario, move) : step2048(scenario, move);
            revision++; renderScene(); syncRequest(); applied = scenario.turns !== before;
            if (!applied) $('scene-message').textContent = 'The model chose a move that changes nothing. Autoplay paused; try again or play manually.';
          }
          if (scenario.over || !applied) stopAuto();
        } else applied = true;
        $('status').textContent = 'Ready';
      } else $('status').textContent = 'Cancelled';
      controls(); job.resolve(applied);
    }
    if (data.type === 'error') {
      const job = pending; pending = undefined; busy = false; stopAuto();
      $('status').textContent = ready ? 'Ready' : 'Load failed'; showError(data.message);
      job?.resolve(false); controls();
    }
    if (data.type === 'log') console.info(data.text);
  };
}
function decide(request, apply, source) {
  if (!ready || busy || pending) return Promise.resolve(false);
  showError(''); busy = true; $('status').textContent = 'Thinking…';
  const id = ++nextId;
  return new Promise(resolve => {
    pending = {id, revision, apply, source, request: structuredClone(request), turn: scenario.turns, scenarioId: scenario.id, resolve};
    controls(); worker.postMessage({type: 'decide', id, request});
  });
}
async function runDemo() {
  try {
    const request = scenario.id === 'custom' ? JSON.parse($('request').value) : requestFor(scenario);
    return await decide(request, isGame(), scenario.id === 'custom' ? 'custom' : 'demo');
  } catch (error) { showError(error.message); return false; }
}
async function autoStep() {
  if (!autoRunning || !ready || scenario.over) return stopAuto();
  const applied = await runDemo();
  if (!autoRunning) return;
  if (!applied || --remaining <= 0) {
    stopAuto();
    if (remaining <= 0) $('scene-message').textContent = '50 model turns completed. Start autoplay again to continue.';
    return;
  }
  autoTimer = setTimeout(autoStep, 280);
}
function selectDemo(id, scroll = false) {
  if (pending || autoRunning) return;
  revision++; scenario = createScenario(id); showError(''); clearResult();
  for (const card of $('gallery').querySelectorAll('button')) {
    const selected = card.dataset.demo === id;
    card.classList.toggle('selected', selected); card.setAttribute('aria-pressed', String(selected));
  }
  renderScene(); syncRequest(); controls();
  if (id === 'custom') $('inspector').open = true;
  if (scroll) $('workspace').scrollIntoView({behavior: 'smooth', block: 'start'});
}
function edited() { revision++; clearResult(); syncRequest(); }
function manualMove(direction) {
  if (busy || autoRunning || scenario.over || !isGame()) return;
  scenario = scenario.id === 'snake' ? stepSnake(scenario, direction) : step2048(scenario, direction);
  edited(); renderScene(); controls();
}
function renderScene() {
  const demo = catalog.find(d => d.id === scenario.id), scene = $('scene');
  const restoreFocus = Boolean(document.activeElement?.closest('.game-surface'));
  scene.replaceChildren(); $('demo-title').textContent = demo.title;
  $('demo-kicker').textContent = isGame() ? 'LIVE GAME · REAL MODEL MOVES' : demo.kind === 'custom' ? 'YOUR OWN EXPERIMENT' : 'LIVE WORKFLOW';
  $('demo-description').textContent = isGame() ? 'You control the state. The game computes each move’s consequences; qev chooses the move.' : demo.description;
  $('demo-count').textContent = isGame() ? `TURN ${scenario.turns}` : '';
  $('new-scene').textContent = isGame() ? 'Reset game' : scenario.id === 'custom' ? 'Reset JSON' : 'Next sample';
  $('demo-run').firstChild.textContent = isGame() ? 'Let the model choose ' : scenario.id === 'custom' ? 'Run this request ' : 'Ask the model ';
  $('scene-message').textContent = isGame() ? scenario.message : '';
  $('demo-hint').textContent = isGame() ? 'Arrow keys work when the board is focused. Autoplay makes up to 50 real model decisions; pause at any time.'
    : scenario.id === 'routing' ? 'The model picks a tool. This demo does not call that tool or connect to your accounts.'
    : scenario.id === 'custom' ? 'Edit the JSON below. Choice, yes/no and ordinal score questions are supported.'
    : 'Choose a sample or edit the text. Every answer comes from the loaded model.';
  if (isGame()) {
    const surface = document.createElement('div'); surface.className = 'game-surface';
    const top = document.createElement('div'); top.className = 'game-topline';
    top.append(text('span', scenario.id === 'snake' ? 'SNAKE / 8 × 8' : '2048 / 4 × 4'), text('strong', `${scenario.score} pts`)); surface.append(top);
    if (scenario.id === 'snake') {
      const board = document.createElement('div'); board.className = `snake-board${scenario.over ? ' over' : ''}`;
      board.tabIndex = 0; board.setAttribute('role', 'group'); board.setAttribute('aria-label', 'Snake board. Arrow keys move the snake.');
      for (let y = 0; y < scenario.size; y++) for (let x = 0; x < scenario.size; x++) {
        const at = p => p[0] === x && p[1] === y;
        const type = at(scenario.body[0]) ? 'head' : scenario.body.some(at) ? 'body' : scenario.walls.some(at) ? 'wall' : at(scenario.food) ? 'food' : 'empty';
        const cell = document.createElement('button'); cell.className = `snake-cell ${type}`;
        cell.tabIndex = -1; cell.setAttribute('aria-label', `Column ${x}, row ${y}, ${type}`); cell.dataset.x = x; cell.dataset.y = y;
        cell.onclick = () => {
          if (busy || autoRunning) return;
          scenario = editSnake(scenario, x, y, editMode); edited(); renderScene(); controls();
        }; board.append(cell);
      }
      surface.append(board);
      const tools = document.createElement('div'); tools.className = 'board-tools';
      const label = text('label', 'Click board to'); const select = document.createElement('select'); select.id = 'edit-mode';
      for (const [value, name] of [['food', 'Move food'], ['wall', 'Toggle walls']]) { const option = text('option', name); option.value = value; select.append(option); }
      select.value = editMode; select.onchange = () => { editMode = select.value; };
      label.append(select); tools.append(label); surface.append(tools);
    } else {
      const board = document.createElement('div'); board.className = 'tiles-board'; board.tabIndex = 0;
      board.setAttribute('role', 'group'); board.setAttribute('aria-label', '2048 board. Arrow keys move tiles.');
      scenario.board.forEach((value, i) => { const tile = text('div', value || '', `tile${value >= 128 ? ' high-tile' : ''}`); tile.dataset.value = value; tile.setAttribute('aria-label', `Row ${Math.floor(i / 4)}, column ${i % 4}, ${value || 'empty'}`); board.append(tile); });
      surface.append(board);
    }
    const manual = document.createElement('div'); manual.className = 'manual-controls'; manual.append(text('span', 'PLAY YOURSELF'));
    for (const d of ['left', 'up', 'down', 'right']) { const button = text('button', arrows[d]); button.dataset.move = d; button.setAttribute('aria-label', `Move ${d}`); button.onclick = () => manualMove(d); manual.append(button); }
    surface.append(manual); scene.append(surface);
    if (restoreFocus) surface.querySelector('[role=group]').focus({preventScroll: true});
  } else if (scenario.id === 'custom') {
    const div = document.createElement('div'); div.className = 'custom-scene';
    div.append(text('strong', '{ state, questions }'), text('p', 'An editable state. Your own questions.\nA probability distribution from the same local model.'));
    scene.append(div);
  } else {
    const card = document.createElement('div'); card.className = 'message-card';
    const heading = document.createElement('div'); heading.className = 'message-heading';
    const symbols = {guardrails: '⌁', inbox: '↳', triage: '?', routing: '↗', moderation: '≋'};
    heading.append(text('span', symbols[scenario.id], 'avatar'));
    const title = document.createElement('div'); title.append(text('strong', scenario.id === 'guardrails' ? 'Untrusted document' : scenario.id === 'inbox' ? 'Incoming email' : scenario.id === 'triage' ? 'Customer ticket' : scenario.id === 'routing' ? 'User message' : 'Community post'), text('span', 'EDITABLE EXAMPLE')); heading.append(title);
    const select = document.createElement('select'); select.className = 'sample-select'; select.setAttribute('aria-label', 'Example sample');
    samples[scenario.id].forEach((sample, i) => { const option = text('option', sample.label); option.value = i; select.append(option); }); select.value = scenario.sample;
    select.onchange = () => { scenario = useSample(scenario, Number(select.value)); edited(); renderScene(); controls(); }; heading.append(select); card.append(heading);
    if (scenario.id === 'inbox') {
      const meta = document.createElement('div'); meta.className = 'email-meta';
      for (const key of ['from', 'subject']) { const label = text('label', key === 'from' ? 'From' : 'Subject'); const input = document.createElement('input'); input.type = 'text'; input.value = scenario[key]; input.oninput = () => { scenario[key] = input.value; edited(); }; label.append(input); meta.append(label); } card.append(meta);
    }
    const input = document.createElement('textarea'); input.className = `scenario-input${scenario.id === 'guardrails' ? ' tall' : ''}`; input.id = 'scenario-text';
    input.value = scenario.text; input.setAttribute('aria-label', `${demo.title} input text`); input.oninput = () => { scenario.text = input.value; edited(); }; card.append(input);
    if (['routing', 'inbox'].includes(scenario.id)) {
      const choices = requestFor(scenario).questions[scenario.id === 'routing' ? 'tool' : 'folder'].criteria;
      const chips = document.createElement('div'); chips.className = 'tool-options';
      Object.keys(choices).forEach(key => { const chip = text('span', key.replaceAll('_', ' '), 'tool-chip'); chip.dataset.choice = key; chips.append(chip); }); card.append(chips);
    }
    scene.append(card); updateChips();
  }
}
function updateChips() {
  const result = lastResult?.revision === revision && lastResult.source !== 'custom' ? lastResult.result : null;
  const selected = result?.answers.tool?.choice || result?.answers.folder?.choice;
  for (const node of $('scene').querySelectorAll('[data-choice]')) node.classList.toggle('chosen', node.dataset.choice === selected);
}
function renderResult(result, workerMs, job) {
  lastResult = {revision, result, source: job.source}; $('answers').replaceChildren();
  const name = catalog.find(d => d.id === job.scenarioId).title;
  $('result-context').textContent = job.source === 'custom' ? 'Custom JSON request · no game action applied.' : job.apply ? `${name} · response for the state before turn ${job.turn + 1}.` : `${name} · real model response for the current text.`;
  for (const [id, answer] of Object.entries(result.answers)) {
    const section = document.createElement('div'); section.className = 'question';
    section.append(text('h3', answer.choice ? `${id} → ${answer.choice.replaceAll('_', ' ')}` : answer.type === 'noul' ? `${id} · ${(answer.noul * 100).toFixed(1)}% yes` : `${id} · score ${answer.score}`));
    const probabilities = answer.probabilities || {no: 1 - answer.noul, yes: answer.noul};
    for (const [key, p] of Object.entries(probabilities)) {
      const winner = answer.choice ? key === answer.choice : answer.type === 'noul' && key === (answer.noul >= .5 ? 'yes' : 'no');
      const row = document.createElement('div'); row.className = `prob${winner ? ' winner' : ''}`;
      const label = document.createElement('div'); label.className = 'prob-label';
      label.append(text('span', answer.legend?.[key] ?? key.replaceAll('_', ' ')), text('span', `${(p * 100).toFixed(2)}%`));
      const bar = document.createElement('div'); bar.className = 'bar'; const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${Math.max(0, Math.min(100, p * 100))}%`;
      bar.append(fill); row.append(label, bar); section.append(row);
    }
    if (answer.confidence !== undefined) section.append(text('p', `Confidence ${(answer.confidence * 100).toFixed(1)}% · Certainty ${(answer.certainty * 100).toFixed(1)}%`, 'answer-meta'));
    $('answers').append(section);
  }
  const metrics = document.createElement('div'); metrics.className = 'metric-grid';
  for (const [value, caption] of [[`${Math.round(workerMs)} ms`, 'Local decision'], [result.metrics.evaluated_tokens, 'Input tokens evaluated'], [`${Math.round(result.metrics.wasm_heap_bytes / 1048576)} MiB`, 'WASM heap · not total RAM']]) {
    const cell = document.createElement('div'); cell.append(text('strong', value), text('span', caption)); metrics.append(cell);
  }
  $('metrics').replaceChildren(metrics); $('raw').textContent = JSON.stringify(result, null, 2);
  $('evaluated-request').textContent = JSON.stringify(job.request, null, 2); updateChips();
}
const previews = {
  snake: '<div class="mini-snake"><span></span><i></i></div>',
  '2048': `<div class="mini-tiles">${[2, 0, 4, 0, 0, 8, 0, 2, 16, 0, 4, 0].map(n => `<span>${n || ''}</span>`).join('')}</div>`,
  guardrails: '<div class="mini-text"><small>UNTRUSTED TEXT</small>“Ignore all previous<br>instructions…”</div>',
  inbox: '<div class="mini-text"><small>INCOMING EMAIL</small>↳ Reply today<br>↳ Receipts & newsletters</div>',
  triage: '<div class="mini-text"><small>SUPPORT TICKET</small>“You charged me twice.<br>Please refund one.”</div>',
  routing: '<div class="mini-tools"><span>⌕</span><span>±</span><span>▦</span></div>',
  moderation: '<div class="mini-text"><small>COMMUNITY POST</small>Useful finding<br>or engagement bait?</div>',
  custom: '<div class="mini-code">{ state,<br>  questions }</div>',
};
for (const demo of catalog) {
  const card = document.createElement('button'); card.className = 'demo-card'; card.dataset.demo = demo.id; card.dataset.kind = demo.kind; card.setAttribute('aria-pressed', 'false');
  card.setAttribute('aria-label', `Open ${demo.title} demo`);
  card.innerHTML = `<div class="card-art" aria-hidden="true">${previews[demo.id]}</div><div class="card-copy"><span class="tag">${demo.tag}</span><h3>${demo.title}</h3><p>${demo.description}</p></div><span class="card-arrow" aria-hidden="true">↗</span>`;
  card.onclick = () => selectDemo(demo.id, true); $('gallery').append(card);
}
document.querySelectorAll('[data-filter]').forEach(button => { button.onclick = () => {
  document.querySelectorAll('[data-filter]').forEach(b => { const on = b === button; b.classList.toggle('active', on); b.setAttribute('aria-pressed', String(on)); });
  for (const card of $('gallery').querySelectorAll('button')) card.hidden = button.dataset.filter !== 'all' && card.dataset.kind !== button.dataset.filter;
}; });
for (const option of $('mode').options) option.disabled = (option.value.endsWith('threads') && !crossOriginIsolated) || (option.value !== 'scalar' && !simdAvailable()) || ((option.value.startsWith('relaxed') || option.value.startsWith('fma')) && !relaxedDotAvailable()) || (option.value.startsWith('fma') && !fusedFmaAvailable());
$('mode').value = [...$('mode').options].find(option => !option.disabled).value;
const cores = navigator.hardwareConcurrency || 1;
$('threads').value = String(cores >= 16 ? 8 : cores >= 8 ? 4 : cores >= 4 ? 2 : 1);
$('load').onclick = () => {
  showError(''); ready = false; busy = true; newWorker(); controls(); $('status').textContent = 'Loading';
  const mode = $('mode').value;
  worker.postMessage({type: 'load', mode, threads: mode.endsWith('threads') ? Number($('threads').value) : 1, context: Number($('context').value), cache: $('cache').checked});
};
$('demo-run').onclick = runDemo;
$('autoplay').onclick = () => { if (autoRunning) stopAuto(true); else { autoRunning = true; remaining = 50; controls(); autoStep(); } };
$('new-scene').onclick = () => {
  if (isGame()) { scenario = createScenario(scenario.id, scenario.seed + 1); edited(); renderScene(); }
  else if (scenario.id === 'custom') { customDraft = JSON.stringify(initialCustomRequest, null, 2); revision++; clearResult(); syncRequest(); }
  else { scenario = useSample(scenario, (scenario.sample + 1) % samples[scenario.id].length); edited(); renderScene(); }
  showError(''); controls();
};
$('request').oninput = () => { customDraft = $('request').value; };
$('run').onclick = async () => { try { await decide(JSON.parse($('request').value), false, 'custom'); } catch (error) { showError(error.message); } };
$('cancel').onclick = () => stopAuto(true);
$('reset').onclick = () => {
  stopAuto(); pending?.resolve(false); pending = undefined; worker?.terminate(); worker = undefined;
  ready = busy = false; $('progress').value = 0; $('status').textContent = 'Not loaded';
  $('engine-info').textContent = ''; $('loading').textContent = 'Engine unloaded. Load again to resume AI decisions.'; controls();
};
$('mode').onchange = controls;
$('clear').onclick = async () => {
  try {
    await caches.delete('qev-local-wasm-v1');
    if (navigator.storage?.getDirectory) { const root = await navigator.storage.getDirectory(); try { await root.removeEntry('qev-local-wasm-v1', {recursive: true}); } catch (error) { if (error.name !== 'NotFoundError') throw error; } }
    $('loading').textContent = 'Browser model cache cleared.';
  } catch (error) { showError(error.message); }
};
document.addEventListener('keydown', event => {
  const direction = {ArrowUp: 'up', ArrowRight: 'right', ArrowDown: 'down', ArrowLeft: 'left'}[event.key];
  if (!direction || !event.target.closest('.game-surface') || event.target.closest('select,input,textarea')) return;
  event.preventDefault(); manualMove(direction);
});
selectDemo('snake');
