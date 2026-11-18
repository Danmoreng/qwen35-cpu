"""One real batch throughput observation for benchmark-inference-seq.ps1."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

from openai import OpenAI, APIConnectionError

p = argparse.ArgumentParser()
p.add_argument("--server", required=True, type=Path)
p.add_argument("--model-dir", required=True, type=Path)
p.add_argument("--threads", required=True)
p.add_argument("--batch-width", required=True)
p.add_argument("--profile-json", required=True, type=Path)
args = p.parse_args()
texts = [
    "wir treffen uns am freitag um neun uhr direkt vor dem bahnhof",
    "der kunde möchte die rechnung diesmal per mail statt mit der post bekommen",
    "bitte bring zur besprechung morgen auch die unterlagen aus dem letzten quartal mit",
    "das paket wurde gestern nachmittag beim empfang im erdgeschoss abgegeben",
    "könntest du mir noch kurz sagen wann der nächste zug nach köln fährt",
    "wenn das wetter mitspielt findet das konzert draußen auf dem marktplatz statt",
    "ich habe die präsentation überarbeitet und stelle sie heute im team vor",
    "frau müller hat angerufen weil ihre bestellung bisher noch nicht angekommen ist",
    "am sonntag besuchen wir meine schwester die seit kurzem in bremen wohnt",
    "stell die kiste bitte neben das fenster damit der eingang frei bleibt",
    "der techniker kommt zwischen zehn und zwölf uhr um den router auszutauschen",
    "wir können den termin verschieben falls es am dienstag zeitlich knapp wird",
    "das protokoll liegt im gemeinsamen ordner direkt neben dem projektplan",
    "für die rückfahrt brauchen wir zwei fahrradkarten und ein ticket für das kind",
    "die nächste probe beginnt eine halbe stunde später als ursprünglich vereinbart",
    "ich melde mich sobald die bestätigung von der buchhaltung vorliegt",
]
system = ("Korrigiere die Groß- und Kleinschreibung sowie die Zeichensetzung im deutschen Transkript. "
          "Bewahre die Wortwahl. Gib ausschließlich den korrigierten Satz aus.")
with socket.socket() as sock:
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
proc = subprocess.Popen([args.server.resolve(), "--model-dir", args.model_dir.resolve(), "--port",str(port),
                         "--threads", args.threads, "--batch-width", args.batch_width],
                        stdout=subprocess.DEVNULL)
try:
    with OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key=os.getenv("QWEN35_API_KEY","local"),
                max_retries=0, timeout=60) as client:
        for _ in range(900):
            if proc.poll() is not None:
                raise RuntimeError("Server exited")
            try:
                model = client.models.list().data[0].id
                break
            except APIConnectionError:
                time.sleep(.1)
        else:
            raise TimeoutError("Server did not start")
        def body(text):
            return dict(model=model,temperature=0,max_completion_tokens=48,
                        messages=[{"role":"system","content":system},{"role":"user","content":text}])
        client.chat.completions.create(**body("das ist die aufwärmrunde"))
        endpoint="/v1/chat/completions"
        rows=[dict(custom_id=str(i),method="POST",url=endpoint,body=body(t)) for i,t in enumerate(texts)]
        start=time.monotonic()
        f=client.files.create(file=("transcripts.jsonl","\n".join(map(json.dumps,rows)).encode(),"application/jsonl"),purpose="batch")
        b=client.batches.create(input_file_id=f.id,endpoint=endpoint,completion_window="24h")
        while b.status in {"validating","in_progress","finalizing"}:
            if time.monotonic()-start>240:
                raise TimeoutError("Batch deadline exceeded")
            time.sleep(.02)
            b=client.batches.retrieve(b.id)
        assert b.status=="completed" and b.request_counts.completed==len(rows)
        results=[json.loads(s) for s in client.files.content(b.output_file_id).text.splitlines()]
        elapsed=time.monotonic()-start
        tokens=sum(line["response"]["body"]["usage"]["completion_tokens"] for line in results)
        output={r["custom_id"]:r["response"]["body"]["choices"][0]["message"]["content"] for r in results}
        assert len(output)==len(rows)
        digest=hashlib.sha256(json.dumps(output,sort_keys=True).encode()).hexdigest()
        args.profile_json.write_text(json.dumps(dict(rows=len(rows),seconds=elapsed,completion_tokens=tokens,
            requests_per_minute=len(rows)*60/elapsed, aggregate_output_tokens_per_second=tokens/elapsed,
            output_sha256=digest),indent=2))
finally:
    proc.kill()
    proc.wait(timeout=15)
