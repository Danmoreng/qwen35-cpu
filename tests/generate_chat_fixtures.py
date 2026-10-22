"""Regenerate text-only golden prompts from the pinned model's Jinja template.

Requires Jinja2 (fixture generation only). No model code is executed.
Usage: python tests/generate_chat_fixtures.py /path/to/tokenizer_config.json
"""
import hashlib
import json
from pathlib import Path
import sys
from jinja2.sandbox import ImmutableSandboxedEnvironment

config = Path(sys.argv[1])
template = json.loads(config.read_text())["chat_template"]
def error(message):
    raise ValueError(message)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = error
render = env.from_string(template)
cases = []
for messages in [
    [{"role": "user", "content": "Hello"}],
    [{"role": "system", "content": " Answer briefly. "}, {"role": "user", "content": " Hi!\n"}],
    [{"role": "user", "content": "First"}, {"role": "assistant", "content": "<think>Old reasoning</think>\n\nAnswer"}, {"role": "user", "content": "Next"}],
    [{"role": "user", "content": "First"}, {"role": "assistant", "content": " Answer ", "reasoning_content": " Thought "}],
    [{"role": "user", "content": "First"}, {"role": "assistant", "content": None}],
    [{"role": "user", "content": [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]}],
    [{"role": "system", "content": ""}, {"role": "user", "content": "\u3000Grüße 👋\u00a0"}],
    [{"role": "user", "content": ""}],
    [{"role": "user", "content": "First"}, {"role": "assistant", "content": "<think>\nThought\n</think>\n\nAnswer"}],
    [{"role": "user", "content": "First"}, {"role": "assistant", "content": "A"}, {"role": "user", "content": "Second"}, {"role": "assistant", "content": "B"}],
]:
    cases.append({"messages": messages, "prompt": render.render(messages=messages, add_generation_prompt=True, enable_thinking=False)})
out = {"source": "Qwen3.5-0.8B tokenizer_config.json shipped with danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 at f6cb5cf04a9094670f9041578a3f4e44b94a1395",
       "template_sha256": hashlib.sha256(template.encode()).hexdigest(), "enable_thinking": False, "cases": cases}
Path(__file__).with_name("fixtures").joinpath("qwen-chat.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
