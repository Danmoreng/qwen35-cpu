import json
import os
import shutil
import struct
import sys
from pathlib import Path

root = Path(sys.argv[1])
target = root / "text"
target.mkdir()
with (root / "model.safetensors").open("rb") as source:
    (size,) = struct.unpack("<Q", source.read(8))
    header = json.loads(source.read(size))
    data_start = 8 + size
    prefix = "trunk.language_model."
    tensors = sorted(
        ((key, value) for key, value in header.items() if key.startswith(prefix)),
        key=lambda tensor: tensor[1]["data_offsets"][0],
    )
    assert len(tensors) == 320
    filtered = {"__metadata__": {"format": "pt"}}
    offset = 0
    for key, value in tensors:
        begin, end = value["data_offsets"]
        length = end - begin
        filtered["model.language_model." + key.removeprefix(prefix)] = {
            **value,
            "data_offsets": [offset, offset + length],
        }
        offset += length
    assert offset == 1504786048
    text = json.dumps(filtered, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 8)
    with (target / "model.safetensors").open("wb", buffering=0) as output:
        output.write(struct.pack("<Q", len(text)) + text)
        for _, value in tensors:
            begin, end = value["data_offsets"]
            while begin < end:
                copied = os.copy_file_range(
                    source.fileno(),
                    output.fileno(),
                    end - begin,
                    offset_src=data_start + begin,
                )
                assert copied > 0
                begin += copied
    with (target / "readout.f32").open("wb") as output:
        for key, shape in (("readout.weight", [255, 1024]), ("readout.bias", [255])):
            value = header[key]
            assert value["dtype"] == "F32" and value["shape"] == shape
            begin, end = value["data_offsets"]
            source.seek(data_start + begin)
            data = source.read(end - begin)
            assert len(data) == end - begin
            output.write(data)
shutil.copyfile(root / "trunk/config.json", target / "config.json")
print(
    "Extracted 320 unchanged BF16 text tensors and the two FP32 readout tensors.",
    flush=True,
)
