# QEV prompt order on 50 German support tickets

The released Labqoat qev-0.8b H128/Q4-G32-DOT4 checkpoint routed 32 of 50
tickets correctly in its native System One layout. Moving the complete
question and all five options ahead of the ticket reduced that to 10 of 50.
In the moved layout the model chose **Sicherheit** for every ticket.

## Paired experiment

We wrote and labeled the 50 German examples before inference. There are ten
per department (access, billing, contract, technical support, security), five
direct requests and five boundary cases per department. Some tickets mention
a resolved issue in another department or ask for one specific next action.
The question gives concrete indications of unauthorized access priority.
This is an authored diagnostic for one routing policy, language and checkpoint.

| Arm | Literal section order | Accuracy | Macro-F1 | Mean probability assigned to the gold label |
| --- | --- | ---: | ---: | ---: |
| Native | Context → Question + Options → Answer | 32/50 (64%) | 0.648 | 0.601 |
| Reordered, cold | Question + Options → Context → Answer | 10/50 (20%) | 0.067 | 0.247 |
| Reordered, cached | Question + Options → Context → Answer | 10/50 (20%) | 0.067 | 0.247 |

The prompt ends with `Answer: (` in all three arms. Each ticket retains the
exact same token IDs, question, option text, separators and A–E mapping across
arms. The runner verifies that the block decomposition of every native prompt
is byte-for-byte identical at the token-ID level to the production System One
renderer. An independent check against the unmodified HTTP endpoint reproduced
the native top choice, all returned probabilities and the input length on all
50 tickets. Both orders use the same selected-row projection, QEV choice
temperature, probability rounding and top-choice code. The cached arm registers
the 147-token prefix through the `Context:` header once and restores it for each
of 50 distinct tickets (167–187 tokens total). Its unrounded selected logits
match the cold reordered logits exactly for every option and ticket.

In the paired comparison 23 previously correct answers became wrong and one
previously wrong answer became correct. Nine were correct in both and 17 were
wrong in both (exact two-sided McNemar p = 0.00000298). Both the direct and
boundary subsets fell from 16/25 to 5/25. The native order also struggled with
this rubric: seven of ten access tickets were sent to security. The 64%
baseline should not be taken as general QEV routing accuracy.

## Option-position diagnostic

After observing the single-class result we reversed the order of all five
options in both layouts and reran the same 50 tickets. Sicherheit moved from
letter E to letter A. The native layout now got 37/50 right; the reordered
layout again chose Sicherheit 50/50 times and scored 10/50. Simply preferring
the last letter cannot explain this result. This sensitivity check uses the
same examples and is not an independent evaluation set.

Ticket T03, for example, says the invoice itself is correct but printing it
opens a blank page in three browsers. The native prompt chose the gold label
**Technik** (0.7113). After the section swap its probability fell to 0.0722
and QEV chose **Sicherheit**. The rubric explicitly gives genuine security
concerns priority. The experiment measures the effect of moving sections
under that rule; it cannot establish which part of training caused the bias.

## Reproduce

Build the optional benchmark with the server dependencies and run from the
repository root with the original calibrated QEV package. The executable uses
the native CPU engine directly. Each JSONL row contains the ticket ID, token
count, five selected logits, calibrated class probabilities, winning label and
restored-token count for all three arms. This experiment measures quality only.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQWEN35_BUILD_SERVER=ON
cmake --build build --target qev_prompt_order --config Release -j4
study=docs/results/qev-ticket-order-2026-11-21
mkdir -p benchmarks/qev-ticket-order
build/qev_prompt_order models/qev "$study/panel.json" 4 > benchmarks/qev-ticket-order/rows.jsonl
python "$study/summarize.py" "$study/panel.json" benchmarks/qev-ticket-order/rows.jsonl
```

For the option-position diagnostic, reverse the insertion order of the five
entries in the panel's `criteria` object and rerun the same commands. This
reassigns option letters while retaining the gold department labels.

[Panel and gold labels](../results/qev-ticket-order-2026-11-21/panel.json) ·
[Paired metrics and confusion matrices](../results/qev-ticket-order-2026-11-21/summary.json) ·
[Per-ticket readouts](../results/qev-ticket-order-2026-11-21/rows.jsonl.gz) ·
[Position diagnostic](../results/qev-ticket-order-2026-11-21/reversed-label-summary.json) ·
[Provenance](../results/qev-ticket-order-2026-11-21/provenance.json)
