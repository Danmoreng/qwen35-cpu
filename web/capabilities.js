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
let fusedFma;
export function fusedFmaAvailable() {
  if (fusedFma !== undefined) return fusedFma;
  try {
    // (f32, f32, f32) -> splat -> relaxed_madd -> extract lane 0.
    const body = [0,32,0,253,19,32,1,253,19,32,2,253,19,253,133,2,253,31,0,11];
    const bytes = new Uint8Array([0,97,115,109,1,0,0,0,
      1,8,1,96,3,125,125,125,1,125,3,2,1,0,
      7,7,1,3,102,109,97,0,0,10,body.length+2,1,body.length,...body]);
    const {fma} = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports;
    const maximum = (2 - 2 ** -23) * 2 ** 127;
    fusedFma = fma(1 + 2 ** -23, 1 - 2 ** -23, -1) === -(2 ** -46) &&
      fma(maximum, 2, -maximum) === maximum &&
      fma(2 ** -126, .5, 2 ** -149) === 2 ** -127 + 2 ** -149 &&
      Object.is(fma(-0, 1, -0), -0);
  } catch { fusedFma = false; }
  return fusedFma;
}
