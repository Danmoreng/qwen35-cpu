"""Exercise a running local server with the official OpenAI Python SDK."""
import argparse
import json
import os
import time

import openai
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
args = parser.parse_args()
client = OpenAI(base_url=args.base_url, api_key=os.getenv("QWEN35_API_KEY", "local"),
                max_retries=0, timeout=90)
model = client.models.list().data[0].id
api = client.responses


def expect_error(error_type, operation):
    try:
        operation()
    except error_type:
        return
    raise AssertionError(f"Expected {error_type.__name__}")


def terminal(response):
    deadline = time.monotonic() + 90
    while response.status in {"in_progress", "queued"}:
        assert time.monotonic() < deadline
        time.sleep(0.05)
        response = api.retrieve(response.id)
    return response


prompt = "Answer with exactly one word. Is water wet?"
instructions = "Reply in English."
settings = dict(model=model, input=prompt, instructions=instructions, temperature=0,
                max_output_tokens=16)
first = api.create(**settings, metadata={"case": "sdk"})
assert first.output_text and first.metadata["case"] == "sdk"
assert first.output_text == api.retrieve(first.id).output_text
count = api.input_tokens.count(input=prompt, instructions=instructions)
assert count.input_tokens == first.usage.input_tokens
chat = client.chat.completions.create(model=model, messages=[
    {"role": "system", "content": instructions}, {"role": "user", "content": prompt}],
    temperature=0, max_completion_tokens=16)
assert first.output_text == chat.choices[0].message.content

with api.stream(**settings) as stream:
    events = list(stream)
    final = stream.get_final_response()
assert final.output_text == first.output_text
assert [event.sequence_number for event in events] == list(range(len(events)))
assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == final.output_text
assert events[-1].type == f"response.{final.status}"

follow = api.create(model=model, previous_response_id=first.id, input="Make it shorter.",
                    max_output_tokens=8, temperature=0)
items = list(api.input_items.list(follow.id, order="asc", limit=1))
assert [item.role for item in items] == ["user", "assistant", "user"]
assert items[0].content[0].text == prompt and items[2].content[0].text == "Make it shorter."
assert follow.instructions is None
stateless = api.create(**settings, store=False)
expect_error(openai.NotFoundError, lambda: api.retrieve(stateless.id))
referenced = api.input_tokens.count(model=model, instructions=instructions,
                                    input=[{"type": "item_reference", "id": items[0].id}])
assert referenced.input_tokens == count.input_tokens
api.delete(first.id)
expect_error(openai.NotFoundError, lambda: api.retrieve(first.id))
assert api.retrieve(follow.id).id == follow.id

schema = {"type": "object", "properties": {"ok": {"type": "boolean", "const": True}},
          "required": ["ok"], "additionalProperties": False}
structured = api.create(model=model, input="Return the requested JSON.", temperature=0.8,
                        max_output_tokens=40, text={"format": {
                            "type": "json_schema", "name": "result", "schema": schema, "strict": True}})
assert structured.status == "completed" and json.loads(structured.output_text) == {"ok": True}
short_json = api.create(model=model, input="Reply with JSON.", max_output_tokens=1,
                        text={"format": {"type": "json_schema", "name": "result", "schema": schema}})
assert short_json.status == "incomplete" and short_json.incomplete_details.reason == "max_output_tokens"
expect_error(openai.BadRequestError, lambda: api.create(
    model=model, input="Reply with JSON.", text={"format": {"type": "json_schema", "name": "result",
                                                            "schema": {"type": "string", "pattern": "abc"}}}))
tool = {"type": "function", "name": "confirm", "parameters": schema, "strict": True}
with api.stream(model=model, input="Call confirm.", tools=[tool], tool_choice="required",
                max_output_tokens=100, temperature=0) as stream:
    function_events = list(stream)
    call_response = stream.get_final_response()
call = call_response.output[0]
assert call.type == "function_call" and call.name == "confirm" and json.loads(call.arguments) == {"ok": True}
assert any(e.type == "response.function_call_arguments.done" for e in function_events)
continued = api.create(model=model, previous_response_id=call_response.id, input=[{
    "type": "function_call_output", "call_id": call.call_id, "output": "The request succeeded."}],
    max_output_tokens=8)
assert continued.output_text
bg = api.create(**settings, background=True)
assert terminal(bg).output_text == first.output_text
with api.create(**settings, background=True, stream=True) as stream:
    created = next(iter(stream))
assert created.type == "response.created"
with api.retrieve(created.response.id, stream=True, starting_after=created.sequence_number) as stream:
    replay = list(stream)
assert replay[0].sequence_number == created.sequence_number + 1
assert replay[-1].response.output_text == first.output_text
long_request = api.create(model=model, input="Write a long story about a lighthouse, with ten chapters.",
                          max_output_tokens=1024, background=True)
cancelled = api.cancel(long_request.id)
assert cancelled.status == "cancelled" and api.retrieve(cancelled.id).status == "cancelled"
limited = api.create(model=model, input="Write a paragraph about the ocean.", max_output_tokens=1, temperature=0)
assert limited.status == "incomplete" and limited.incomplete_details.reason == "max_output_tokens"
expect_error(openai.BadRequestError, lambda: api.create(**settings, top_logprobs=5))
expect_error(openai.BadRequestError, lambda: api.create(**settings, tools=[{"type": "web_search"}]))
expect_error(openai.BadRequestError, lambda: api.create(**settings, store=False, background=True))
# A preceding long user turn can be dropped without cutting messages or the current question.
old = {"role": "user", "content": " apples" * 9000}
latest = {"role": "user", "content": prompt}
truncated = api.create(model=model, input=[old, {"role": "assistant", "content": "OK."}, latest],
                       truncation="auto", instructions=instructions, temperature=0, max_output_tokens=16)
assert truncated.usage.input_tokens == count.input_tokens and truncated.output_text == first.output_text
expect_error(openai.BadRequestError, lambda: api.create(model=model, input=[old], truncation="auto"))
expect_error(openai.BadRequestError, lambda: api.create(model=model, input=[old, latest], truncation="disabled"))
print(f"OpenAI {openai.__version__} Responses SDK checks passed")
