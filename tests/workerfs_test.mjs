import assert from 'node:assert/strict';
import {bufferWorkerFSReads} from '../web/workerfs.js';

const pageBytes = 64;
const data = Uint8Array.from({length: 521}, (_, i) => (i * 31 + 7) % 256);
const node = {name: 'model.q35h', size: data.length, data};
const other = {name: node.name, size: 91, data: new Uint8Array(91).fill(42)};
let reads = 0;
const originalRead = function(stream, buffer, offset, length, position) {
  assert.equal(this, workerFS.stream_ops);
  reads++;
  const bytes = stream.node.data.subarray(position, position + length);
  buffer.set(bytes, offset); return bytes.length;
};
const workerFS = {stream_ops: {read: originalRead}};
const progress = [];
const restore = bufferWorkerFSReads(workerFS, {pageBytes, onRead: (...args) => progress.push(args)});
function check(file, position, length) {
  const target = new Uint8Array(length + 8).fill(255);
  const count = workerFS.stream_ops.read({node: file}, target, 3, length, position);
  const expected = file.data.subarray(position, position + length);
  assert.equal(count, expected.length);
  assert.deepEqual(target.subarray(3, 3 + count), expected);
  assert.ok(target.subarray(0, 3).every(x => x === 255));
  assert.ok(target.subarray(3 + count).every(x => x === 255));
}
// Sequential short reads must match the file across page boundaries while
// reducing storage calls to one per page, including the partial final page.
for (let position = 0; position < data.length; position += 7) check(node, position, 7);
assert.equal(reads, Math.ceil(data.length / pageBytes));
const beforeEOF = reads;
check(node, data.length, 10); check(node, data.length + 10, 10); check(node, 0, 0);
assert.equal(reads, beforeEOF);
// Backward seeks, interleaved files with the same name and direct large reads.
for (const [file, position, length] of [[node, 60, 13], [other, 0, 15], [node, 63, 9],
  [node, 100, 200], [node, 500, 100], [other, 88, 9]]) check(file, position, length);
assert.ok(progress.some(([name, position, size]) => name === node.name && position === size));
restore();
assert.equal(workerFS.stream_ops.read, originalRead);
assert.throws(() => bufferWorkerFSReads(workerFS, {pageBytes: 0}));
assert.throws(() => bufferWorkerFSReads(workerFS, {pageBytes: 1.5}));
// Storage errors remain visible and the caller can restore the reader on failure.
workerFS.stream_ops.read = () => { throw new Error('Storage read failed'); };
const restoreFailure = bufferWorkerFSReads(workerFS, {pageBytes});
assert.throws(() => workerFS.stream_ops.read({node}, new Uint8Array(1), 0, 1, 0), /Storage read failed/);
restoreFailure();
console.log('WORKERFS buffering: storage-call bounds, exact bytes, seeks, EOF, file isolation and errors passed.');
