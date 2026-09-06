#!/usr/bin/env python3
"""Freeze a small independent synthetic audit; never use it for fitting/seeds."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path('.cache/python-deps').resolve()))
from transformers import AutoTokenizer

out = Path('benchmarks/head-audit-v1')
out.mkdir(parents=True, exist_ok=False)
tokenizer = AutoTokenizer.from_pretrained('models/hf-download-test', local_files_only=True)
english = '''A coastal observatory replaced its temperature sensor during a winter storm.
The replacement reported values two degrees higher than the old instrument. The technician
did not immediately conclude that the air had warmed. She checked the installation height,
the shelter around the sensor, and the time stamps in both logs. For three days the devices
operated beside each other. Their readings agreed at night but diverged after sunrise.
This pattern suggested that sunlight was heating one enclosure. The team moved both sensors
into identical shelters and repeated the comparison. They preserved the original records
and documented the equipment change so that future analysts could distinguish a measurement
artifact from a change in the local climate.'''
german = '''Ein kleines Theater plante eine Vorstellung ohne elektrisches Licht. Die Gruppe
probte am Nachmittag, um die Bewegung der Schatten zu beobachten. Zuerst standen alle Stühle
in geraden Reihen. Von den hinteren Plätzen war die Bühne jedoch kaum zu erkennen. Deshalb
stellten die Helfer die Stühle in einem Halbkreis auf und ließen einen breiten Gang frei.
Während der zweiten Probe öffnete jemand die Seitentür. Ein Windstoß bewegte den Vorhang,
und die Schauspielerin unterbrach ihren Satz. Später wurde die Tür geschlossen gehalten.
Die Leitung schrieb jede Änderung in ein Heft. Am Abend verglich sie die Beobachtungen mit
dem ursprünglichen Plan und entschied, welche Lösungen bei der nächsten Probe bleiben sollten.'''
math = '\n'.join(f'A box contains {n} red balls and {n+3} blue balls. There are {2*n+3} balls in total. '
    f'The probability of drawing a red ball is {n}/{2*n+3}. After a red ball is removed without replacement, '
    f'the next red probability is {n-1}/{2*n+2}. The product gives the probability of two red draws.' for n in range(3,12))
code = '\n'.join(f'def moving_average_{n}(values):\n    width = {n}\n    if len(values) < width:\n        return []\n'
    '    total = sum(values[:width])\n    result = [total / width]\n'
    '    for i in range(width, len(values)):\n        total += values[i] - values[i-width]\n'
    '        result.append(total / width)\n    return result\n' for n in range(2,80))
chat = tokenizer.apply_chat_template([
    dict(role='user', content='How can I organize a small community seed exchange?'),
    dict(role='assistant', content='Choose a date and an accessible room. Ask participants to label each packet with the plant name, '
         'harvest year, and growing conditions. Put empty envelopes and pencils on a table. Explain that people can contribute '
         'seeds, take seeds, or simply ask questions. Reserve a separate area for plants with uncertain identification. '
         'After the event, record which seeds remain and arrange a dry storage place. A volunteer can collect feedback '
         'about the location and timing before planning another exchange.')], tokenize=False, add_generation_prompt=False)
documents = [('english-prose', english, 32), ('german-prose', german, 32),
             ('mathematics', math, 64), ('code', code, 128),
             ('rendered-chat', chat, 32), ('long-recurrent-code', code, 4096)]
old = json.loads(Path('benchmarks/plan-heldout-v1/quality-windows.json').read_text())
old_ids = {w['article_sha256'] for w in old['windows']}
windows = []
for i, (domain, text, prompt) in enumerate(documents):
    ids = tokenizer.encode(text, add_special_tokens=False)
    identity = hashlib.sha256(text.encode()).hexdigest()
    assert identity not in old_ids and len(ids) >= prompt+64
    folder = out/str(i)
    folder.mkdir()
    (folder/'source.txt').write_text(text, encoding='utf-8')
    for name, values in [('prompt', ids[:prompt]), ('targets', ids[prompt:prompt+64])]:
        (folder/(name+'.csv')).write_text(','.join(map(str, values)))
    windows.append(dict(window=i, domain=domain, folder=str(folder), prompt_tokens=prompt,
        scored_tokens=64, article_sha256=identity, scoring_mask=[True]*64))
(out/'quality-windows.json').write_text(json.dumps(dict(version=1,
    scope='384 positions; authored synthetic audit, not general capability certification. Code windows share one document.',
    calibration_usage='Excluded from calibration, quantizer fitting and seed selection',
    overlap_check='Document SHA256 disjoint from six development documents; code windows intentionally share source',
    tokenizer_sha256=hashlib.sha256(Path('models/hf-download-test/tokenizer.json').read_bytes()).hexdigest(),
    windows=windows), indent=2)+'\n')
