"""Check template inheritance and override precedence without loading model weights."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
spec = spec_from_file_location("import_finetune", Path(sys.path[0]) / "import-finetune.py")
importer = module_from_spec(spec)
spec.loader.exec_module(importer)

with TemporaryDirectory() as tmp:
    base, adapter, out = (Path(tmp) / name for name in ("base", "adapter", "out"))
    for directory in (base, adapter, out):
        directory.mkdir()
    importer.write(base / "tokenizer_config.json", {"chat_template": "base"})
    importer.write(adapter / "tokenizer_config.json", {"eos_token": "<end>"})
    importer.copy_model_assets(adapter, base, out)
    assert importer.read(out / "tokenizer_config.json") == {"chat_template": "base", "eos_token": "<end>"}

    (base / "chat_template.jinja").write_text("base file")
    (base / "chat_templates").mkdir()
    (base / "chat_templates/tool_use.jinja").write_text("base tool file")
    importer.write(adapter / "tokenizer_config.json", {"chat_template": "adapter"})
    importer.copy_model_assets(adapter, base, out)
    assert importer.read(out / "tokenizer_config.json")["chat_template"] == "adapter"
    assert not (out / "chat_template.jinja").exists() and not (out / "chat_templates").exists()

    (adapter / "chat_templates").mkdir()
    (adapter / "chat_templates/tool_use.jinja").write_text("adapter tool")
    (adapter / "chat_template.jinja").write_text("adapter default")
    importer.copy_model_assets(adapter, base, out)
    assert (out / "chat_template.jinja").read_text() == "adapter default"
    assert (out / "chat_templates/tool_use.jinja").read_text() == "adapter tool"
