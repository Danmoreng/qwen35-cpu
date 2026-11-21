# Historical work

Superseded comparisons, optimization diaries, review prompts, session handoffs
and their one-off result generators were removed from the current tree during
the documentation cleanup. They remain available in the immutable
[pre-cleanup snapshot](https://github.com/Danmoreng/qwen35-cpu/tree/3680efb33bf6975ce6722d64d20c641cb7490b06).
No Git history was rewritten.

- [Historical reports](https://github.com/Danmoreng/qwen35-cpu/tree/3680efb33bf6975ce6722d64d20c641cb7490b06/docs)
  include the extraction baseline, earlier CPU comparisons, selective Q8/G16
  experiments and native optimization investigations.
- [Historical evidence](https://github.com/Danmoreng/qwen35-cpu/tree/3680efb33bf6975ce6722d64d20c641cb7490b06/docs/results)
  preserves the previously committed raw runs, manifests, checksums and source snapshots.
- [Historical scripts](https://github.com/Danmoreng/qwen35-cpu/tree/3680efb33bf6975ce6722d64d20c641cb7490b06/scripts)
  preserve the generators for those fixed experiment layouts.
- The first Intel/Ryzen comparison used eight workers on the i7 and six regenerated
  GGUF files. The original report and evidence are preserved in Git history.
  [Report](https://github.com/Danmoreng/qwen35-cpu/blob/23a89ca34a26fae45ba10f7144df153e7b26d170/docs/benchmarks/i7-8750h-vs-ryzen-9955hx3d.md) · [Evidence](https://github.com/Danmoreng/qwen35-cpu/tree/23a89ca34a26fae45ba10f7144df153e7b26d170/docs/results/i7-ryzen-comparison-2026-11-20)

Use that snapshot as a whole when reproducing an old experiment. Its commands,
paths and assumptions need not match current source. Large raw captures that
were never tracked remain local under ignored `benchmarks/`; a manifest is not
a substitute for those captures.

The current [documentation index](README.md) links the supported workflows.
The [evidence inventory](results/README.md) retains the compact data behind
current published claims, including model-packaging inputs.
