#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  echo 'Usage: download-model.sh OWNER/MODEL COMMIT_SHA [DESTINATION]' >&2; exit 1
fi
repo="$1"; revision="$2"; destination="${3:-models/qwen3.5-0.8b}"
[[ "$repo" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || { echo 'Invalid model repository' >&2; exit 1; }
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo 'Supply an immutable 40-character model commit' >&2; exit 1; }
[[ ! -e "$destination" ]] || { echo 'Choose a new destination directory' >&2; exit 1; }
mkdir -p -- "$destination"
cd -- "$destination"
base="https://huggingface.co/$repo/resolve/$revision"
curl --fail --location --retry 3 "$base/SHA256SUMS" --output SHA256SUMS
files=(model.q35h config.json tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja LICENSE NOTICE README.md quantization.json)
[[ $(wc -l < SHA256SUMS) -eq ${#files[@]} ]] || { echo 'Unexpected checksum manifest' >&2; exit 1; }
for file in "${files[@]}"; do
  [[ $(grep -Ec "^[0-9a-f]{64}  ${file//./\.}$" SHA256SUMS) -eq 1 ]] || { echo "Missing/duplicate checksum: $file" >&2; exit 1; }
  curl --fail --location --retry 3 "$base/$file" --output "$file.part"
  expected=$(awk -v name="$file" '$2 == name {print $1}' SHA256SUMS)
  actual=$(sha256sum "$file.part"); actual=${actual%% *}
  [[ "$actual" == "$expected" ]] || { echo "Checksum mismatch: $file" >&2; exit 1; }
  mv -- "$file.part" "$file"
done
echo "Verified model in $destination"
