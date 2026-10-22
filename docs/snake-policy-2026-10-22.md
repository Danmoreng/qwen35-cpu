# Snake policy regression, 2026-10-22

The local Snake demo repeatedly circled the upper-right corner without eating.
This was reproducible with the unchanged H128/Q4-G32-DOT4 checkpoint in the
actual in-app WASM browser: zero food after 40 turns, with the final moves
repeating the same four-cell cycle. The previous prompt used one `choice`
question with null option descriptions and numeric Manhattan-distance facts.

## Change

Snake now asks an independent `noul` question per non-reversing direction, as
in the original qev demo. The UI applies the largest model yes score, keeps the
raw responses intact, and explains that these scores do not sum to 100%.
Invalid/incomplete scores stop autoplay. No model-selected move is replaced by
a heuristic or safety filter.

The game also supplies explicitly **path-assisted** navigation facts. For each
candidate, breadth-first search finds a shortest open path around walls and
the post-move body, held fixed. The text distinguishes a shortest available
route, a longer detour, no currently open route, eating, and immediate death.
These computed facts substantially assist navigation: this is not an unaided
model spatial-reasoning benchmark. Tail motion, growth, future survival and
arbitrary edited boards are not solved by this static search.

Full-state repetition detection pauses autoplay instead of allowing an
indefinite cycle. It compares heading, ordered body, target and walls, with
history reset on food collection or board edits. Normal head-position revisits
are allowed. The 50-turn autoplay limit remains unchanged.

## Functional results

| Actual browser, same initial board and seed | Turns | Food collected | Collision / repeated full state |
|---|---:|---:|---|
| Previous Snake policy | 40 | 0 | Repeated corner cycle |
| Updated path-assisted policy | 40 | 5 | Neither |
| Updated path-assisted policy, continued | 100 | 13 | Neither |

The continued run included 50 uninterrupted autoplay moves from turn 40 to
turn 90, stopping at the configured limit with 12 food collected. Manual AI
steps completed turns 91–100. Recorded responses had zero decode forwards and
zero output tokens. Browser console warnings/errors were empty.

Four additional 100-turn runs through the native browser C ABI used initial
food positions `[5,2]`, `[0,0]`, `[7,7]` and `[0,7]`: **13, 12, 11 and 12** food
items, respectively, without collision or full-state repetition. Each complete
request and selected move was replay-checked against the final production
`requestFor`, `selectSnakeMove` and `stepSnake` helpers (400 decisions).
Spawning is deterministic from seed 7251 but depends on the evolving board;
different policies need not encounter identical later food positions.

These are small deterministic regression checks, not a general win rate,
quantization-quality evaluation or performance comparison. The model, weights,
calibration, tokenizer and inference kernels did not change. Three yes/no
questions require more inference work per turn than the previous single choice
question; no speed improvement is claimed.

## Exploration and verification

Clearer natural-language move consequences improved the single-choice version
to seven food items in 80 turns, but longer runs could still loop. Absolute
path distances alone did not fix this. Qualitative shortest-route/detour facts
combined with separate yes/no questions performed best in this small study;
the same facts with a single choice question performed worse. This is why the
final patch changes both the facts and the question format. An initial
exploratory yes/no selector read a nonexistent response field; those prototype
rows are excluded. Subsequent studies used `answer.noul`, and the final helper
was checked against every selected move in the four complete runs.

`node tests/demo_scenarios_test.mjs` passes, including obstacle detours,
disconnected targets, tail-vacating, immediate eating, independent-score
selection, malformed scores, full-state repetition and history reset. The
browser integration script now checks food collection over 30 AI turns; it was
syntax-checked, with actual UI validation performed in the in-app browser.

Configuration, hashes and compact results are preserved in
[`results/qev-snake-policy-2026-10-22/checks.json`](results/qev-snake-policy-2026-10-22/checks.json).
Raw local studies, requests, responses, the small stdin-driven native oracle
and the screenshot are under `benchmarks/qev-snake-fix-2026-10-22/` (ignored).
The native oracle links the existing browser API library and is only a
functional test driver. Native checks used 8 threads, chunk 32, context 2048 and
the native default FP16 KV; the browser used `fma-threads`, 8 threads, chunk 32,
context 2048 and FP32 KV. These runs were unpinned and their incidental timings
must not be used for speed claims.

All changes remain local. Nothing was published or deployed.
