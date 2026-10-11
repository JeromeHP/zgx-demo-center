# Porting to the HP ZGX Nano

**Done.** This was written before the Nano was available; the plan below was executed on
2026-09-18 and the demo now runs there as Demo Center demo id `grant` on port 8095. What
follows is what actually happened, with the original llama.cpp/vLLM routes kept at the bottom
as fallbacks that were not needed.

The app needed **one** code change to move -- not zero, as this file originally predicted. The
prediction was right that hardware is a config concern, and wrong that the *server* is: Ollama
and llama-server disagree about how to name and control reasoning, and the disagreement is
silent. See "What actually changed" below.

## What was done

1. **Preflight.** `uname -m` → `aarch64`, GB10 Blackwell, 121GB memory, Docker present,
   8095 free. Ollama already running on 11434.

2. **Model server: none built.** The Nano's Ollama already served `qwen3.6:35b` (22GB) --
   the same model family this app was built against. No llama.cpp build, no GGUF download, no
   second GPU consumer competing with the other Demo Center demos for VRAM. `llm.base_url`
   points at `http://127.0.0.1:11434/v1`.

3. **Endpoint check, `--label nano-ollama`, 3 reps:** E1-E12 all PASS
   (`reports/endpoint_check_nano-ollama_*.json`). Against
   `reports/baseline_zbook.json`:

   | | ZBook (ROCm llama.cpp) | Nano (Ollama, GB10) |
   |---|---|---|
   | E3 reasoning source | `field` | `field` |
   | E11 decode | 23.8 tok/s | **77.2 tok/s (3.2x)** |
   | E11 TTFT | ~0.69 s | ~0.10 s |
   | E10 usage | present | present |

4. **Python runtime.** No `uv` on the Nano; the app runs under the shared `zgx` conda env
   (Python 3.12.13) like every other native demo. That env already had 13 of 14 dependencies
   at the required versions. See `docs/DEMO_CENTER_INTEGRATION.md`.

5. **App container: not used.** The Nano runs the demo as a host process via
   `start_demo_8095.sh`. `docker/compose.yaml` still works and is kept as the portable path.

6. **E2E, quick scope.** This is the number that answers the question this file was written to
   ask -- *does faster hardware solve the live-demo timing problem?*

   **Yes.** See `PROGRESS.md` Phase 15 for the measured medians against the ZBook's 1313 s
   (~21.9 min) and the original <=360 s target.

7. **Demo Center integration.** Done -- `docs/DEMO_CENTER_INTEGRATION.md`.

## What actually changed in the code

`app/llm_client.py` gained two things, both because Ollama's OpenAI-compatible layer differs
from llama-server's in ways that fail *silently*:

- It reads the separated-reasoning field under either name (`reasoning_content` for
  llama-server, `reasoning` for Ollama). Without this the UI's thinking panel is simply empty.
- A new `llm.thinking_control` setting selects `chat_template_kwargs.enable_thinking`
  (llama-server, vLLM -- the default, so that path is unchanged) or `reasoning_effort`
  (Ollama, which ignores the former entirely and honours `"none"` as the only real off switch).

Neither is hardware-specific; both are server-specific, and both are covered by tests.

## Fallbacks that were not needed

**Option A: llama.cpp CUDA with the same GGUF.**
```bash
git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(nproc)" --target llama-server
```
Then run `scripts/start_model_server.sh` with `LLAMA_SERVER_BIN` pointed at the new binary,
and set `llm.thinking_control: chat_template`.

**Option B: vLLM + NVFP4.** `nvidia/Qwen3.6-35B-A3B-NVFP4`, preferring NVIDIA's vLLM container
for DGX Spark/GB10 if available (model card recommends vLLM >=0.19.0):
```bash
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen3.6-35b-a3b \
  --host 127.0.0.1 --port 8080 \
  --max-model-len 65536 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --language-model-only
```
Re-run the endpoint check with `--label nano-vllm-nvfp4`. Watch E3 (reasoning source -- and
read the `reasoning_source` value in the JSON, not just PASS/FAIL; see
`docs/DEMO_CENTER_INTEGRATION.md` for why E3 can pass vacuously), E7/E8 (nested and streaming
tool-call argument assembly), and E10 (`usage` in streamed responses).

Either option would take VRAM outside the Demo Center's `/ollama_release` protocol, which is
the main reason neither was chosen.
