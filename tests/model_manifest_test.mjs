import assert from 'node:assert/strict';
import {validateModelManifest, modelSha256, modelRepository, modelFiles} from '../web/model-manifest.js';
const local = {format: 'qev-local-wasm/1', model_sha256: modelSha256,
  files: modelFiles.map(name => ({name, bytes: 100, sha256: modelSha256}))};
assert.equal(validateModelManifest(local), local);
const revision = 'eb167df11cbd6130513fedfcfdfb2ca11180b2de';
const remote = {...local, repository: modelRepository, revision,
  files: local.files.map(file => ({...file, url: `https://huggingface.co/${modelRepository}/resolve/${revision}/${file.name}`}))};
assert.equal(validateModelManifest(remote), remote);
for (const change of [m => m.revision = 'main', m => m.repository = 'another/model',
  m => m.files[0].url = 'https://example.com/model.q35h', m => m.files[1].url = undefined,
  m => m.files[0].sha256 = '0'.repeat(64), m => m.files.pop(),
  m => m.files[1].name = '../config.json', m => m.files[1].name = 'model.q35h',
  m => m.files[1].bytes = -1]) {
  const bad = structuredClone(remote); change(bad); assert.throws(() => validateModelManifest(bad));
}
console.log('Local/remote manifests, immutable URLs, checkpoint identity and file bounds passed.');
