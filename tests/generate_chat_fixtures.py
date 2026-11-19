"""Regenerate text-only golden prompts from the pinned model's Jinja template.

Requires Jinja2 (fixture generation only). No model code is executed.
Usage: python tests/generate_chat_fixtures.py /path/to/tokenizer_config.json
"""
import hashlib
import json
from pathlib import Path
import sys
from copy import deepcopy
from jinja2.sandbox import ImmutableSandboxedEnvironment

config = Path(sys.argv[1])
file = config.with_name("chat_template.jinja")
template = file.read_text(encoding="utf-8") if file.exists() else json.loads(config.read_text(encoding="utf-8"))["chat_template"]
def error(message):
    raise ValueError(message)
env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
env.globals["raise_exception"] = error
env.filters["tojson"] = lambda value, **kwargs: json.dumps(value, ensure_ascii=False, **kwargs)
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
tool = {"type": "function", "function": {"name": "measure", "description": "Maß für Δ", "parameters": {
    "type": "object", "properties": {"name": {"type": "string"}, "scale": {"type": "number"},
    "flags": {"type": "array", "items": {"type": "boolean"}}},
    "required": ["name", "scale", "flags"], "additionalProperties": False}}}
args = {"name": "Größe", "scale": 1.0, "flags": [True, False]}
history = [
    {"role": "system", "content": " Be precise. "},
    {"role": "user", "content": "Measure Ω."},
    {"role": "assistant", "content": None, "reasoning_content": "  Use the meter.  ", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": "measure", "arguments": args}}]},
    {"role": "tool", "content": " 12.5 ", "tool_call_id": "call_1"},
]
wire = deepcopy(history)
wire[2]["tool_calls"][0]["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
cases.append({"messages": wire, "prompt": render.render(messages=history, tools=[tool], add_generation_prompt=True),
              "tools": [tool]})
messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
cases.append({"messages": messages, "prompt": render.render(messages=messages, enable_thinking=True, add_generation_prompt=True),
              "chat_template_kwargs": {"enable_thinking": True}})
out = {"source": "Qwen3.5-0.8B tokenizer_config.json shipped with danmoreng/Qwen3.5-0.8B-H128-Q4-G32-DOT4 at f6cb5cf04a9094670f9041578a3f4e44b94a1395",
       "template_sha256": hashlib.sha256(template.encode()).hexdigest(), "enable_thinking": False, "cases": cases}
Path(__file__).with_name("fixtures").joinpath("qwen-chat.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding='utf-8')
