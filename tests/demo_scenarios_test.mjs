import assert from 'node:assert/strict';
import {catalog, samples, createScenario, requestFor, useSample, snakeOutcome,
  snakeDirections, stepSnake, editSnake, move2048, step2048} from '../web/scenarios.js';

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
assert.equal(requestFor(snake).questions.move.type, 'choice');
assert.deepEqual(Object.keys(requestFor(snake).questions.move.criteria), snakeDirections(snake));
console.log('Demo scenarios: merge rules/directions, deterministic spawning, Snake collisions/tail/growth/editing, immutable previews and request contracts passed.');
