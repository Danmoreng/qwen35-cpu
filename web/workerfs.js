// C++ streams issue small reads. Buffer immutable WORKERFS files so each read
// does not require a synchronous browser-storage round trip. Keep only one page.
export function bufferWorkerFSReads(workerFS, {pageBytes = 1024 * 1024, onRead} = {}) {
  if (!Number.isSafeInteger(pageBytes) || pageBytes <= 0) throw new Error('Invalid WORKERFS page size');
  const originalRead = workerFS.stream_ops.read;
  let cachedNode, page, pageStart = 0;
  workerFS.stream_ops.read = function(stream, buffer, offset, length, position) {
    const node = stream.node;
    const available = Math.min(length, Math.max(0, node.size - position));
    if (available <= 0) return 0;
    let copied = 0;
    if (available >= pageBytes) {
      copied = originalRead.call(this, stream, buffer, offset, available, position);
    } else {
      while (copied < available) {
        const cursor = position + copied;
        if (cachedNode !== node || cursor < pageStart || cursor >= pageStart + page.length) {
          pageStart = Math.floor(cursor / pageBytes) * pageBytes;
          const bytes = new Uint8Array(Math.min(pageBytes, node.size - pageStart));
          const count = originalRead.call(this, stream, bytes, 0, bytes.length, pageStart);
          cachedNode = node; page = bytes.subarray(0, count);
          if (cursor >= pageStart + count) break;
        }
        const start = cursor - pageStart;
        const count = Math.min(available - copied, page.length - start);
        buffer.set(page.subarray(start, start + count), offset + copied);
        copied += count;
      }
    }
    onRead?.(node.name, position + copied, node.size);
    return copied;
  };
  return () => {
    workerFS.stream_ops.read = originalRead;
    cachedNode = page = undefined;
  };
}
