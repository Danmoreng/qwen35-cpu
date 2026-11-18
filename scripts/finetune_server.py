"""Local server lifetime for fine-tune quality comparisons."""
from contextlib import contextmanager
import os
import socket
import subprocess
import time
import urllib.request


@contextmanager
def running_server(package, server, threads, extra=()):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    command = [server, "--model-dir", str(package), "--port", str(port), "--threads", str(threads),
               "--batch-width", "1", "--prefix-cache", "off", *extra]
    child = subprocess.Popen(command, env={**os.environ, "QWEN35_API_KEY": ""})
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Evaluation server exited before readiness")
            try:
                with urllib.request.urlopen(url + "/health", timeout=1):
                    break
            except OSError:
                time.sleep(.1)
        else:
            raise TimeoutError("Evaluation server readiness deadline")
        yield url
    finally:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
