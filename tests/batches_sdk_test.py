"""Run official OpenAI SDK batch/file contracts against a local native model."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import urllib.error
import subprocess
import tempfile
import time
from urllib.request import Request, urlopen

import openai
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--server", type=Path, required=True)
parser.add_argument("--model-dir", type=Path, required=True)
parser.add_argument("--qev-model-dir", type=Path)
args = parser.parse_args()
key = "batch-test-key"


@contextmanager
def server(database, *options, model_dir=None):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}/v1"
    command = [str(args.server.resolve()), "--model-dir", str((model_dir or args.model_dir).resolve()),
               "--port", str(port), "--batch-db", str(database), *options]
    process = subprocess.Popen(command, env={**os.environ, "QWEN35_API_KEY": key})
    try:
        with OpenAI(base_url=base, api_key=key, max_retries=0, timeout=30) as client:
            for _ in range(600):
                if process.poll() is not None:
                    raise RuntimeError("Server exited")
                try:
                    model = client.models.list().data[0].id
                    break
                except openai.APIConnectionError:
                    time.sleep(.1)
            else:
                raise RuntimeError("Server failed to start")
            with urlopen(Request(base+"/batch-config", headers={"Authorization": "Bearer "+key})) as r:
                tuning = json.load(r)
            yield client, model, tuning
    finally:
        process.kill()
        process.wait(timeout=15)


def upload(client, rows):
    data = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows).encode()
    return client.files.create(file=("input.jsonl", data, "application/jsonl"), purpose="batch")


def submit(client, rows, endpoint):
    return client.batches.create(input_file_id=upload(client, rows).id, endpoint=endpoint,
                                 completion_window="24h", metadata={"suite": "native"})


def wait(client, batch):
    deadline = time.monotonic()+120
    while batch.status not in {"completed", "failed", "cancelled", "expired"}:
        assert time.monotonic() < deadline
        time.sleep(.04)
        batch = client.batches.retrieve(batch.id)
    return batch


def lines(client, file_id):
    return [json.loads(line) for line in client.files.content(file_id).text.splitlines()]


def row(index, endpoint, **body):
    return dict(custom_id=str(index), method="POST", url=endpoint, body=body)


with tempfile.TemporaryDirectory() as tmp:
    db = Path(tmp)/"jobs.sqlite"
    chat = "/v1/chat/completions"
    with server(db) as (client, model, tuning):
        assert tuning["probes"] and 1 <= tuning["threads"] <= tuning["cpu_capacity"]
        print("AUTOTUNE", json.dumps(tuning), flush=True)
        try:
            urlopen(str(client.base_url)+"batches")
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        else:
            raise AssertionError("Batch endpoints require the configured API key")
        large = b'{"custom_id":"large"}\n' * 16000
        large_file = client.files.create(file=("large.jsonl", large), purpose="batch")
        assert client.files.content(large_file.id).content == large
        assert client.files.delete(large_file.id).deleted
        sample = dict(model=model, messages=[{"role":"user","content":"Name the capital of France."}],
                      temperature=0, max_completion_tokens=12)
        ref = client.chat.completions.create(**sample).choices[0].message.content
        batch = wait(client, submit(client, [row(0, chat, **sample), row(1, chat, invalid=True)], chat))
        assert batch.status == "completed" and batch.request_counts.completed == batch.request_counts.failed == 1
        output = lines(client, batch.output_file_id)[0]
        assert output["custom_id"] == "0" and output["response"]["body"]["choices"][0]["message"]["content"] == ref
        assert lines(client, batch.error_file_id)[0]["response"]["status_code"] == 400
        invalid = wait(client, submit(client, [row(0, chat, **sample)]*2, chat))
        assert invalid.status == "failed" and invalid.errors.data[0].line == 2
        assert {f.id for f in client.files.list(purpose="batch_output",limit=1)} >= {batch.output_file_id, batch.error_file_id}
        assert {b.id for b in client.batches.list(limit=1)} >= {batch.id, invalid.id}
        assert client.files.delete(batch.error_file_id).deleted
        try:
            client.files.retrieve(batch.error_file_id)
        except openai.NotFoundError:
            pass
        else:
            raise AssertionError("Deleted file remains visible")
        for endpoint, body in [("/v1/responses",dict(input="Say hello.",max_output_tokens=8)),
                               ("/v1/completions",dict(prompt="The capital of France is",max_tokens=8))]:
            b=wait(client,submit(client,[row(0,endpoint,model=model,temperature=0,**body)],endpoint))
            assert b.status=="completed"
            expected = "response" if endpoint=="/v1/responses" else "text_completion"
            assert lines(client,b.output_file_id)[0]["response"]["body"]["object"] == expected
        cancel = submit(client, [row(i, chat, **sample) for i in range(24)], chat)
        assert wait(client, client.batches.cancel(cancel.id)).status == "cancelled"
        resumed = submit(client,[row(i,chat,**sample) for i in range(12)],chat)
        # Interrupt while there are both committed and unfinished rows.
        deadline=time.monotonic()+60
        while True:
            b=client.batches.retrieve(resumed.id)
            if 0<b.request_counts.completed<12:
                break
            assert time.monotonic()<deadline and b.status!="completed"
            time.sleep(.02)
        baseline_file = batch.output_file_id
    with server(db, "--threads", "2", "--batch-width", "1") as (client, model, tuning):
        assert tuning["threads"] == 2 and tuning["batch_width"] == 1 and not tuning["probes"]
        assert lines(client,baseline_file)[0] == output
        b=wait(client,client.batches.retrieve(resumed.id))
        restored=lines(client,b.output_file_id)
        assert b.request_counts.completed==12 and len(restored)==12
        assert {x["custom_id"] for x in restored} == {str(i) for i in range(12)}
        assert all(x["response"]["body"]["choices"][0]["message"]["content"]==ref for x in restored)
    if args.qev_model_dir:
        with server(Path(tmp)/"qev.sqlite", model_dir=args.qev_model_dir) as (client, model, tuning):
            endpoint = "/v1/systemone"
            question = {"instructions": "Who should handle this ticket?", "type": "choice",
                        "criteria": {"billing": "payment and refunds", "tech": "software problems"}}
            body = {"state": {"ticket": "I was charged twice for the same order."},
                    "questions": {"department": question}}
            request = Request(str(client.base_url)+"systemone", data=json.dumps(body).encode(),
                              headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"})
            with urlopen(request) as r:
                expected = json.load(r)["answers"]
            result = wait(client,submit(client,[row(i,endpoint,**body) for i in range(3)],endpoint))
            assert result.request_counts.completed == 3
            assert all(item["response"]["body"]["answers"] == expected for item in lines(client,result.output_file_id))
            print("QEV",json.dumps(tuning),flush=True)
print(f"OpenAI {openai.__version__} batch/file and recovery checks passed")
