import assert from 'node:assert/strict';
import {catalog, samples, createScenario, requestFor, useSample, snakeOutcome,
  snakeDirections, snakeMoveFacts, selectSnakeMove, stepSnake, editSnake, move2048, step2048} from '../web/scenarios.js';

// Each tile merges at most once. Directions must not leak across row boundaries.
const row = values => [...values, ...Array(12).fill(0)];
assert.deepEqual(move2048(row([2, 2, 2, 2]), 'left').board.slice(0, 4), [4, 4, 0, 0]);
assert.deepEqual(move2048(row([2, 2, 4, 0]), 'left').board.slice(0, 4), [4, 4, 0, 0]);
assert.equal(move2048(row([2, 2, 2, 2]), 'left').points, 8);
assert.deepEqual(move2048(row([2, 0, 2, 4]), 'right').board.slice(0, 4), [0, 0, 4, 4]);
const column = [2, 0, 0, 0, 2, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0];
assert.deepEqual(move2048(column, 'up').board, [4, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
assert.deepEqual(move2048(column, 'down').board, [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 8, 0, 0, 0]);
const blocked = [2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2];
for (const direction of ['up', 'right', 'down', 'left']) assert.equal(move2048(blocked, direction).changed, false);
let board = createScenario('2048', 123), copy = structuredClone(board);
const after = step2048(board, 'left');
assert.deepEqual(board, copy); // Request generation / previews do not advance RNG or mutate the board.
assert.deepEqual(after, step2048(copy, 'left'));
assert.equal(after.turns, 1);
assert.ok([2, 4].includes(after.board.reduce((a, b) => a + b, 0) - board.board.reduce((a, b) => a + b, 0)));
const noMove = {...board, board: blocked};
assert.equal(step2048(noMove, 'left').turns, 0);

let snake = createScenario('snake'); copy = structuredClone(snake);
assert.equal(snakeDirections(snake).includes('left'), false);
assert.equal(stepSnake(snake, 'left').turns, 0);
assert.equal(snakeOutcome({...snake, walls: [[4, 4]]}, 'right').hit, 'wall');
const eat = {...snake, food: [4, 4]};
const eaten = stepSnake(eat, 'right');
assert.equal(eaten.score, 10); assert.equal(eaten.body.length, 4);
assert.ok(![...eaten.body, ...eaten.walls].some(p => JSON.stringify(p) === JSON.stringify(eaten.food)));
const tail = {...snake, heading: 'up', body: [[2, 2], [2, 3], [1, 3], [1, 2]], food: [7, 7], walls: []};
assert.equal(snakeOutcome(tail, 'left').hit, null); // The tail vacates this square.
assert.equal(stepSnake({...snake, walls: [[4, 4]]}, 'right').over, true);
assert.deepEqual(editSnake(snake, 3, 4, 'food'), snake); // No food on the snake.
assert.deepEqual(editSnake(snake, 5, 2, 'wall'), snake); // Do not overwrite food.
const edited = editSnake(snake, 7, 7, 'wall'); assert.equal(edited.walls.length, snake.walls.length + 1);
assert.deepEqual(snake, copy);

for (const {id} of catalog.filter(d => d.kind !== 'custom')) {
  const state = createScenario(id), before = structuredClone(state), request = requestFor(state);
  assert.deepEqual(state, before);
  assert.equal(request.encoding, 'per_question');
  assert.ok(Object.keys(request.questions).length > 0);
  for (const q of Object.values(request.questions)) assert.ok(['choice', 'noul', 'score'].includes(q.type));
  if (samples[id]) for (let i = 0; i < samples[id].length; i++) assert.ok(requestFor(useSample(state, i)).state);
}
// Route facts must account for obstacles, rather than just Manhattan distance.
const detour = {...snake, size: 5, body: [[1, 2], [1, 3], [1, 4]], heading: 'up', food: [3, 2], walls: [[2, 0], [2, 1], [2, 2], [2, 3]]};
const facts = snakeMoveFacts(detour);
assert.match(facts.right, /dies/);
assert.match(facts.left, /shortest open route, 7 steps/);
assert.match(facts.up, /longer detour/);
const enclosed = {...detour, walls: [...detour.walls, [2, 4]]};
assert.match(snakeMoveFacts(enclosed).left, /no path.*body held fixed/);
assert.match(snakeMoveFacts(eat).right, /eats the food safely/);
assert.match(snakeMoveFacts(tail).left, /route|detour|no path/); // Vacating tail is a legal candidate.
const snakeRequest = requestFor(snake);
assert.deepEqual(Object.keys(snakeRequest.questions), snakeDirections(snake));
assert.ok(Object.values(snakeRequest.questions).every(q => q.type === 'noul'));
const scores = {up: {type: 'noul', noul: .9}, right: {type: 'noul', noul: .95}, down: {type: 'noul', noul: .1}};
assert.equal(selectSnakeMove(snakeRequest, scores), 'right'); // Independent scores need not sum to one.
assert.equal(selectSnakeMove(snakeRequest, {...scores, right: {type: 'noul', noul: NaN}}), null);
assert.equal(selectSnakeMove(snakeRequest, {...scores, right: {type: 'noul', noul: 2}}), null);
assert.equal(selectSnakeMove(snakeRequest, {up: scores.up}), null);
assert.equal(selectSnakeMove(snakeRequest, {...scores, right: scores.up}), 'up'); // Stable tie order.
// Repeated head positions alone are insufficient: compare body, heading and target too.
let circle = {...createScenario('snake'), walls: [], food: [7, 7]};
for (const d of ['right', 'down', 'left', 'up']) circle = stepSnake(circle, d);
assert.equal(circle.looped, false);
for (const d of ['right', 'down', 'left', 'up']) circle = stepSnake(circle, d);
assert.equal(circle.looped, true);
const retargeted = editSnake(circle, 0, 0, 'food');
assert.equal(retargeted.looped, false); assert.deepEqual(retargeted.recentStates, []);
assert.equal(eaten.looped, false); assert.equal(eaten.recentStates.length, 1);
console.log('Demo scenarios: merge rules/directions, deterministic spawning, Snake collisions/tail/growth/editing, immutable previews and request contracts passed.');
