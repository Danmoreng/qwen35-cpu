// Only the published qev checkpoint is supported by this browser runtime.
export const modelRepository = 'danmoreng/qev-0.8b-H128-Q4-G32-DOT4';
export const modelSha256 = 'e7d61e7b191de6c2afb8663e8a1f8c5f4d52315e9492db7a690708d744f8dc69';
export const modelFiles = ['model.q35h', 'config.json', 'tokenizer.json',
  'tokenizer_config.json', 'vocab.json', 'merges.txt', 'qev_config.json'];

export function validateModelManifest(manifest) {
  if (manifest?.format !== 'qev-local-wasm/1' || manifest.model_sha256 !== modelSha256
      || !Array.isArray(manifest.files) || manifest.files.length !== modelFiles.length)
    throw new Error('Unsupported qev model manifest');
  const names = new Set();
  const remote = manifest.files.some(file => file.url !== undefined);
  if (remote && (manifest.repository !== modelRepository || !/^[0-9a-f]{40}$/.test(manifest.revision)))
    throw new Error('Remote model must use the pinned public qev repository');
  for (const file of manifest.files) {
    if (!modelFiles.includes(file.name) || names.has(file.name)
        || !Number.isSafeInteger(file.bytes) || file.bytes <= 0 || !/^[0-9a-f]{64}$/.test(file.sha256))
      throw new Error('Invalid model file manifest');
    names.add(file.name);
    if (file.name === 'model.q35h' && file.sha256 !== modelSha256)
      throw new Error('Unexpected qev checkpoint hash');
    if (remote && file.url !== `https://huggingface.co/${modelRepository}/resolve/${manifest.revision}/${file.name}`)
      throw new Error('Remote model URL does not match its immutable revision');
  }
  return manifest;
}
