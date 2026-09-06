#!/usr/bin/env python3
"""Prepare a versioned, local cross-domain screening suite (not a population benchmark)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--model-dir', default='models/hf-download-test')
args = parser.parse_args()
sys.path.insert(0, str(Path('.cache/python-deps').resolve()))
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
german = '''Am Morgen fuhr Lea mit dem Zug in eine kleine Stadt am Fluss. Sie wollte im Archiv
alte Karten ansehen und herausfinden, wie sich das Ufer verändert hatte. Auf der ältesten Karte
führte ein schmaler Weg direkt zur Mühle. Heute steht dort eine Bibliothek. Der Archivar erklärte,
dass die Mühle nach einem Hochwasser abgerissen worden war. Lea verglich die Jahreszahlen mit
den Einträgen in einem Notizbuch. Nicht alle Angaben passten zusammen: Ein Bericht beschrieb
eine Brücke, die auf der entsprechenden Karte noch fehlte. Sie notierte beide Quellen, ohne den
Widerspruch aufzulösen. Am Nachmittag ging sie zum Fluss und fotografierte die verbliebenen
Steinmauern. Das Wasser stand niedrig, sodass die Fundamente sichtbar waren. Erst der Vergleich
zwischen Karten, Berichten und dem heutigen Gelände würde eine zuverlässige Beschreibung
ermöglichen. Auf dem Rückweg beschloss sie, bei ihrem nächsten Besuch auch die Bauakten zu lesen.'''
math_text = '\n'.join(f'For x = {n}, the square is x*x = {n*n}. The difference between consecutive '
    f'squares is ({n}+1)^2 - {n}^2 = 2*{n}+1 = {2*n+1}. '
    f'This follows by expanding the product and cancelling the common quadratic term.' for n in range(1, 20))
chat = tokenizer.apply_chat_template([
    dict(role='user', content='How would you investigate two historical sources that disagree?'),
    dict(role='assistant', content='I would first check their dates, authors, and purposes. A map can omit a structure '
         'that already existed, while a later report may describe an earlier event from memory. I would record both '
         'claims and search for independent evidence such as building records or photographs. If that evidence does '
         'not resolve the disagreement, the final account should state what remains uncertain. A precise date should '
         'not be invented merely to produce a tidy chronology.'),
    dict(role='user', content='What information should I record for each source?'),
    dict(role='assistant', content='Record its title, author, creation date, archive identifier, and the exact claim you '
         'are examining. Separate the date of the document from the date of the event. Include a quotation or a '
         'careful transcription when you have permission, and note missing pages, ambiguous handwriting, and later '
         'annotations. Keep your interpretation separate from what the source actually says.')],
    tokenize=False, add_generation_prompt=False)
documents = [('german-prose', german, 64, 128),
             ('english-prose', Path('README.md').read_text(encoding='utf-8'), 256, 256),
             ('code', Path('src/weights/gguf.cpp').read_text(encoding='utf-8'), 256, 256),
             ('mathematics', math_text, 128, 128), ('rendered-chat', chat, 64, 128),
             ('long-recurrent-code', Path('src/runtime/weights.inl').read_text(encoding='utf-8'), 4096, 128)]
for domain, text, prompt, scored in documents:
    if len(tokenizer.encode(text, add_special_tokens=False)) < prompt + scored:
        raise ValueError(f'{domain}: insufficient source tokens')
args.out.mkdir(parents=True, exist_ok=False)
windows = []
for index, (domain, text, prompt, scored) in enumerate(documents):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < prompt + scored:
        raise ValueError(f'{domain}: insufficient source tokens ({len(ids)})')
    folder = args.out/str(index)
    folder.mkdir()
    for name, tokens in [('prompt', ids[:prompt]), ('targets', ids[prompt:prompt+scored])]:
        (folder/(name+'.csv')).write_text(','.join(map(str, tokens)))
    windows.append(dict(window=index, domain=domain, folder=str(folder), prompt_tokens=prompt,
                        scored_tokens=scored, article_sha256=hashlib.sha256(text.encode()).hexdigest()))
(args.out/'quality-windows.json').write_text(json.dumps(dict(version=1,
    scope='Local cross-domain screening; six documents, not general quality certification',
    calibration_usage='Excluded from calibration and quantizer fitting', windows=windows), indent=2)+'\n')
