# Local live decision playground

The local page at `http://127.0.0.1:8765/` now offers seven interactive demos
plus a custom System One JSON editor. The gallery is inspired by the
[original qev examples](https://app.labqoat.com/e/qev/qev/), with independent
local UI and game implementations. No videos, original site code or remote
inference endpoints are embedded. Nothing was deployed or published.

| Demo | Interaction | Model question |
|---|---|---|
| Snake | Manual arrow keys/buttons, move food, toggle walls, AI step, autoplay | Choose a direction from computed move consequences |
| 2048 | Manual arrow keys/buttons, AI step, autoplay, reset | Choose a direction from legal movement/merge/space facts |
| Guardrails | Edit untrusted text or switch samples | Probability of prompt injection |
| Inbox | Edit sender, subject and body; switch samples | Choose an inbox folder |
| Support desk | Edit a customer ticket; switch samples | Team, urgency and ordinal sentiment |
| Tool routing | Edit a user message; switch samples | Choose a tool; no tool is actually called |
| Moderation | Edit a community post; switch samples | Engagement bait, vague hype and useful information |

One shared worker/model serves the selected demo. Games are playable manually
without loading the model. Every AI step and autoplay turn waits for a real
response from the browser's existing H128/Q4-G32-DOT4 WASM engine. Autoplay
pauses on request, game over, an ineffective move or after 50 turns. It never
substitutes a heuristic move for an unsuitable model choice.

As on the original page, the game computes action consequences before asking
the model. Here each turn uses one `choice` question. The original Snake sheet
shows separate yes/no questions per direction; this UI does not claim those
independent probabilities are the same as a choice distribution. Snake facts
describe immediate collisions and Manhattan food distance, not a path-search
agent. 2048 facts describe deterministic slides/merges before the next random
tile. Seeded spawning makes rule checks reproducible.

The inspector shows the next generated request. Each completed response also
retains its exact evaluated request, so a probability distribution for a
previous game state is distinguishable from the next turn's input. Custom JSON
requests never move the board or label an unrelated workflow's input. State
edits clear obsolete results. Cancellation discards a pending result rather
than applying a late game action; an already completed move is not undone.

The initial set is text-only. Receipts/screenshots requiring an image encoder,
Pac-Man, Tetris and Breakout have not been ported. No GPU backend, model weights,
tokenizer, calibration, inference kernels or public benchmark tables changed.

## Local use

The existing WASM build script copies `web/*` into `dist/wasm`, including the new
`scenarios.js`. For an already built runtime, refresh static assets with
`cp web/* dist/wasm/`, then serve using `python3 scripts/serve-wasm.py --port 8765`.
Reload the page after updates. Load once, select a tile and run a decision.
Focus a game board to use arrow keys. Browser caching remains optional.

## Validation

`node tests/demo_scenarios_test.mjs` checks directional 2048 merges, single-merge
semantics, score/tile conservation, deterministic spawning, Snake collision,
tail-vacating and growth rules, board edits, immutable previews and requests.
Existing capability and FP32 parity-helper tests also pass.

The real in-app browser loaded the unchanged checkpoint and completed all
seven demos with zero decode forwards. Additional manual UI checks covered:

- Real changes in guardrail/moderation responses after switching input samples.
- AI-selected game moves, autoplay and pause, and both games' arrow-key controls.
- Moving Snake food, adding a wall and the model choosing from the updated facts.
- A long custom request cancelled while running; no application error.
- Custom routing output kept separate from the workflow's original input.
- Square board geometry, a 390-pixel viewport without horizontal overflow,
  and no browser console errors or warnings during the final checks.

The existing `tests/wasm_browser_ui.mjs` integration suite was updated for the
new gallery/controls and expanded with game/autoplay checks. Current end-to-end
validation used the actual in-app browser; the headless integration script was
syntax-checked, not rerun in this pass. Screenshots and interactive results are
saved locally under `benchmarks/qev-live-demos-2026-10-22/`.
Displayed interactive timings are not controlled performance measurements.
