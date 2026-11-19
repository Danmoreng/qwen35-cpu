#!/usr/bin/env node
import assert from "node:assert/strict";
import { pathToFileURL } from "node:url";

const [playwright, executablePath, file, stageArg, screenshot] = process.argv.slice(2);
const { chromium } = await import(pathToFileURL(playwright).href);
const stage = Number(stageArg);
const browser = await chromium.launch({ executablePath, args: ["--disable-dev-shm-usage"] });

function game(page) {
  const cell = (row, col) => page.locator(`#cell-${row}-${col}`);
  const state = () => page.locator('button[id^="cell-"][aria-pressed="true"]').evaluateAll(
    items => items.map(item => item.id).sort());
  const expectState = async (...ids) => assert.deepEqual(await state(), ids.sort());
  const expectMoves = async count => {
    const text = await page.locator("#moves").innerText();
    assert.equal(Number(text.match(/\d+/)?.[0]), count);
  };
  const expectWin = async expected => {
    const status = await page.locator("#status").innerText();
    assert.equal(/\bwon\b/i.test(status), expected);
  };
  const center = middle => [
    `cell-${middle - 1}-${middle}`, `cell-${middle}-${middle - 1}`,
    `cell-${middle}-${middle}`, `cell-${middle}-${middle + 1}`, `cell-${middle + 1}-${middle}`
  ];
  return { page, cell, state, expectState, expectMoves, expectWin, center };
}

const checks = [
  [1, "initial", async ({page, expectState, expectMoves, center}) => {
    assert.equal(await page.locator('button[id^="cell-"][aria-pressed]').count(), 25);
    await expectState(...center(2));
    await expectMoves(0);
  }],
  [1, "edge", async ({cell, expectState, expectMoves, center}) => {
    await cell(0, 0).click();
    await expectState(...center(2), "cell-0-0", "cell-1-0", "cell-0-1");
    await expectMoves(1);
  }],
  [1, "win", async ({cell, expectState, expectMoves, expectWin}) => {
    await cell(2, 2).click();
    await expectState();
    await expectMoves(1);
    await expectWin(true);
    await cell(0, 0).evaluate(button => button.click());
    await expectState();
    await expectMoves(1);
  }],
  [1, "reset", async ({page, cell, expectState, expectMoves, expectWin, center}) => {
    await cell(2, 2).click();
    await page.locator("#reset").click();
    await expectState(...center(2));
    await expectMoves(0);
    await expectWin(false);
  }],
  [2, "undo", async ({page, cell, expectState, expectMoves, center}) => {
    const undo = page.locator("#undo");
    assert.equal(await undo.isDisabled(), true);
    await cell(0, 0).click();
    await undo.click();
    await expectState(...center(2));
    await expectMoves(0);
    assert.equal(await undo.isDisabled(), true);
    await cell(0, 0).click();
    await page.locator("#reset").click();
    assert.equal(await undo.isDisabled(), true);
  }],
  [2, "undo-win", async ({page, cell, expectState, expectMoves, expectWin, center}) => {
    await cell(2, 2).click();
    await expectWin(true);
    await page.locator("#undo").click();
    await expectWin(false);
    await expectState(...center(2));
    await expectMoves(0);
    await cell(2, 2).click();
    await expectWin(true);
  }],
  [3, "keyboard", async ({page, cell, expectMoves, expectState, center}) => {
    await cell(0, 0).focus();
    await page.keyboard.press("ArrowLeft");
    assert.equal(await page.evaluate(() => document.activeElement.id), "cell-0-0");
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowDown");
    assert.equal(await page.evaluate(() => document.activeElement.id), "cell-1-1");
    await expectMoves(0);
    await expectState(...center(2));
  }],
  [3, "keyboard-activation", async ({page, cell, expectMoves, expectWin}) => {
    await cell(0, 0).focus();
    await page.keyboard.press("Space");
    await expectMoves(1);
    await page.locator("#reset").click();
    await cell(2, 2).focus();
    await page.keyboard.press("Enter");
    await expectMoves(1);
    await expectWin(true);
  }],
  [4, "resize", async ({page, cell, expectMoves, expectState, center}) => {
    const size = page.locator("#size");
    assert.equal(await size.inputValue(), "5");
    assert.deepEqual(await size.locator("option").evaluateAll(items => items.map(item => item.value).sort()), ["3", "5", "7"]);
    await cell(0, 0).click();
    await size.selectOption("3");
    assert.equal(await page.locator('button[id^="cell-"]').count(), 9);
    await expectState(...center(1));
    await expectMoves(0);
    assert.equal(await page.locator("#undo").isDisabled(), true);
    await size.selectOption("7");
    assert.equal(await page.locator('button[id^="cell-"]').count(), 49);
    await expectState(...center(3));
  }],
  [4, "resize-reset", async ({page, cell, expectMoves, expectState, center}) => {
    const size = page.locator("#size");
    await size.selectOption("7");
    await cell(6, 6).click();
    await page.locator("#reset").click();
    assert.equal(await size.inputValue(), "7");
    await expectMoves(0);
    await expectState(...center(3));
  }],
  [4, "resize-win", async ({page, cell, expectWin, expectState, center}) => {
    await page.locator("#size").selectOption("3");
    await cell(1, 1).click();
    await expectWin(true);
    await page.locator("#undo").click();
    await expectWin(false);
    await expectState(...center(1));
  }],
  [4, "resize-keyboard", async ({page, cell}) => {
    await page.locator("#size").selectOption("3");
    await cell(2, 2).focus();
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowDown");
    assert.equal(await page.evaluate(() => document.activeElement.id), "cell-2-2");
    await page.keyboard.press("ArrowUp");
    assert.equal(await page.evaluate(() => document.activeElement.id), "cell-1-2");
  }]
];

const results = {};
try {
  for (const [firstStage, name, run] of checks) {
    if (firstStage > stage) continue;
    const page = await browser.newPage({viewport: {width: 900, height: 760}, offline: true});
    page.setDefaultTimeout(800);
    try {
      await page.goto(pathToFileURL(file).href, {timeout: 2500});
      await run(game(page));
      results[name] = {pass: true};
    } catch (error) {
      results[name] = {pass: false, reason: error.message.split("\n").slice(0, 5).join("\n")};
    } finally {
      await page.close();
    }
  }
  if (screenshot) {
    const page = await browser.newPage({viewport: {width: 900, height: 760}, offline: true});
    await page.goto(pathToFileURL(file).href, {timeout: 2500});
    await page.screenshot({path: screenshot, timeout: 3000});
  }
} finally {
  await browser.close();
}
process.stdout.write(JSON.stringify(results) + "\n");
