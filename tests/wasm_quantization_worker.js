importScripts('./wasm_quantization_test.js');
createQuantTestModule({locateFile:path=>new URL(path,location.href).href,
  print:log=>postMessage({log}),printErr:log=>postMessage({log}),
  onExit:status=>postMessage({status}),onAbort:reason=>postMessage({status:1,reason})
}).catch(error=>postMessage({status:1,error:String(error)}));
