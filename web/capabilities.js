// A minimal function returning v128.const, validated without executing it.
export function simdAvailable() {
  return WebAssembly.validate(new Uint8Array([
    0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,
    10,22,1,20,0,253,12,...Array(16).fill(0),11
  ]));
}
export function relaxedDotAvailable() {
  const zero = [253,12,...Array(16).fill(0)];
  return WebAssembly.validate(new Uint8Array([
    0,97,115,109,1,0,0,0,1,5,1,96,0,1,123,3,2,1,0,
    10,61,1,59,0,...zero,...zero,...zero,253,147,2,11
  ]));
}
