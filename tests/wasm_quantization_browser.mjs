// Run the same exhaustive quantizer checks in the actual target browser.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
const args=Object.fromEntries(process.argv.slice(2).reduce((a,x,i,all)=>{if(i%2===0)a.push([x.replace(/^--/,''),all[i+1]]);return a;},[]));
const {chromium}=await import(pathToFileURL(path.resolve(args['playwright-module'])).href);
const browser=await chromium.launch({executablePath:args.browser});
try {
  const page=await browser.newPage(); await page.goto('http://127.0.0.1:8765');
  const logs=await page.evaluate(mode=>new Promise((resolve,reject)=>{
    const worker=new Worker(`/review/quantization/${mode}/worker.js`);const logs=[];
    const timer=setTimeout(()=>{worker.terminate();reject(Error('Quantizer test timed out'));},60000);
    worker.onmessage=({data})=>{
      if(data.log)logs.push(data.log);
      if(data.status!==undefined){clearTimeout(timer);worker.terminate();data.status===0?resolve(logs):reject(Error(JSON.stringify(data)));}
    };
    worker.onerror=e=>{clearTimeout(timer);worker.terminate();reject(Error(e.message));};
  }),args.mode || 'fma-threads');
  assert.ok(logs.some(x=>x.includes('exact scalar parity')));
  const result={browser:browser.version(),logs,status:'passed'};
  if(args.output)fs.writeFileSync(args.output,JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify(result));
} finally {await browser.close();}
