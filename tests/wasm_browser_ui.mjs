// Local integration test: real browser, unchanged checkpoint, cache and UI.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
const args = Object.fromEntries(process.argv.slice(2).reduce((out, x, i, all) => {
  if (i % 2 === 0) out.push([x.replace(/^--/, ''), all[i + 1]]); return out;
}, []));
const {chromium} = await import(pathToFileURL(path.resolve(args['playwright-module'])).href);
const browser = await chromium.launch({...(args.browser ? {executablePath: args.browser} : {})});
const requests = [], errors = [], results = [], checks = [];
try {
  const page = await browser.newPage({viewport: {width: 1400, height: 1200}});
  page.on('request', request => requests.push(request.url()));
  page.on('pageerror', error => errors.push(error.message));
  page.on('console', event => { if (event.text().includes('Cache')) console.log(event.text()); });
  await page.goto(args.url || 'http://127.0.0.1:8765');
  const expectedMode = await page.evaluate(async () =>
    (await import('./capabilities.js')).fusedFmaAvailable() ? 'fma-threads' : 'relaxed-threads');
  assert.equal(await page.locator('#mode').inputValue(), expectedMode);
  checks.push(`default runtime: ${expectedMode}`);
  assert.equal(await page.locator('#cache').isChecked(), true);
  await page.locator('#load').click();
  await page.waitForFunction(() => document.getElementById('status').textContent === 'Ready', null, {timeout: 180000});
  assert.equal(await page.locator('[data-demo]').count(),8);
  const cell=await page.locator('.snake-cell').first().boundingBox();
  assert.ok(Math.abs(cell.width-cell.height)<1,'Snake cells stay square');
  checks.push('seven live demos plus custom editor; square board');
  async function decide(demo, key) {
    await page.locator(`[data-demo="${demo}"]`).click();
    await page.locator('#demo-run').click();
    await page.waitForFunction(() => document.getElementById('status').textContent === 'Ready', null, {timeout: 180000});
    const result = JSON.parse(await page.locator('#raw').textContent());
    assert.ok(result.answers[key]);
    assert.equal(result.metrics.decode_forwards, 0); assert.equal(result.usage.output_tokens, 0);
    assert.ok(JSON.parse(await page.locator('#evaluated-request').textContent()).questions[key]);
    assert.equal(await page.locator('#error').textContent(),'');
    results.push(result.answers); checks.push(demo);return result;
  }
  await decide('snake','move');assert.equal(await page.locator('#demo-count').textContent(),'TURN 1');
  await decide('2048','move');assert.equal(await page.locator('#demo-count').textContent(),'TURN 1');
  for(const [demo,key] of [['guardrails','injection'],['inbox','folder'],['triage','team'],['routing','tool'],['moderation','useful']])await decide(demo,key);
  await page.locator('[data-demo="snake"]').click();
  await page.locator('#autoplay').click();
  await page.waitForFunction(()=>Number(document.getElementById('demo-count').textContent.split(' ')[1])>=2);
  await page.locator('#autoplay').click();
  await page.locator('#demo-run:not([disabled])').waitFor();
  const paused=await page.locator('#demo-count').textContent();
  await page.waitForTimeout(800);assert.equal(await page.locator('#demo-count').textContent(),paused);
  checks.push('autoplay pause stops future state changes');
  const out = args.output || 'benchmarks/qev-wasm-local/ui'; fs.mkdirSync(out, {recursive: true});
  await page.screenshot({path: `${out}/desktop.png`, fullPage: true});
  await page.setViewportSize({width: 390, height: 844});
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
  await page.screenshot({path: `${out}/mobile.png`, fullPage: true}); checks.push('responsive layout');
  await page.setViewportSize({width: 1400, height: 1200});
  await page.locator('#inspector').evaluate(e=>{e.open=true;});
  await page.locator('#request').fill(fs.readFileSync('docs/results/qev-api-2026-10-21/long-request.json', 'utf8'));
  await page.locator('#run').click(); await page.locator('#cancel').click();
  await page.waitForFunction(() => document.getElementById('status').textContent === 'Cancelled', null, {timeout: 180000}); checks.push('cancel in UI');
  await page.locator('#reset').click();
  await page.locator('#mode').selectOption('relaxed');
  await page.locator('#load').click();
  await page.waitForFunction(() => document.getElementById('status').textContent === 'Ready', null, {timeout: 180000});
  await decide('inbox', 'folder');
  assert.equal(requests.filter(url => url.includes('/model/model.q35h?')).length, 1); checks.push('model cache reused after unload');
  await page.locator('#inspector').evaluate(e=>{e.open=true;});
  await page.locator('#request').fill('{'); await page.locator('#run').click();
  assert.ok((await page.locator('#error').textContent()).length > 0); checks.push('malformed editor JSON');
  assert.ok(requests.every(url => new URL(url).hostname === '127.0.0.1')); checks.push('all resources from localhost');
  const fallback = await browser.newPage();
  await fallback.addInitScript(() => {
    WebAssembly.Instance = class { constructor() { throw new Error('FMA unavailable'); } };
  });
  await fallback.goto(args.url || 'http://127.0.0.1:8765');
  assert.equal(await fallback.locator('#mode').inputValue(), 'relaxed-threads');
  assert.equal(await fallback.locator('option[value="fma-threads"]').evaluate(option => option.disabled), true);
  assert.equal(await fallback.locator('option[value="fma"]').evaluate(option => option.disabled), true);
  await fallback.close(); checks.push('strict FMA fallback when probe fails');
  assert.deepEqual(errors, []);
  fs.writeFileSync(`${out}/checks.json`, JSON.stringify({checks, results, errors, browser: browser.version()}, null, 2) + '\n');
  console.log(JSON.stringify({checks, errors}));
} finally { await browser.close(); }
