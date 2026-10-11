import sys
from pathlib import Path
from huggingface_hub import snapshot_download

DEST = str(Path.home() / "demos" / "pii-demo-containerized-docker" / "models")
print("Downloading curtburk/pii-masking-demo-models -> %s" % DEST, flush=True)
path = snapshot_download(
    repo_id="curtburk/pii-masking-demo-models",
    local_dir=DEST,
    # Only the fine-tuned detector GGUF — the "base" side is now a stock
    # qwen2.5:32b `ollama pull`, not the old TinyLlama GGUF from this repo.
    allow_patterns=["*pii_detector*.gguf"],
)
print("DONE ->", path, flush=True)
