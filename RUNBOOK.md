# HP ZGX Demo Center — Run Notes (this device)

Operational notes for running the Demo Center on **`ZGX Nano`** (GB10, aarch64,
Ubuntu). Launched **on demand** — it does **not** auto-start and does **not**
need to survive reboots. Accessed **remotely** (no monitor attached).

The suite also runs on AMD ROCm hardware (Strix Halo / gfx1151). The installer
detects the accelerator and installs the matching variant; `python3
demos/zgx_hardware.py` prints what this box was detected as, and the launcher's
system bar shows the same values live. See `docs/HARDWARE_PROFILES.md`.

- Repo location: `~/zgx-demo-center/demos/`  (all scripts run from here)
- **Required symlink:** `~/demos` → `~/zgx-demo-center/demos` (scripts hardcode `~/demos`; see below)
- **Known issues:** see [`KNOWN-ISSUES.md`](KNOWN-ISSUES.md) — check it first when
  something behaves oddly; it is the register of things we already know are wrong.

---

## Setup delta — the `~/demos` symlink

The repo lives at `~/zgx-demo-center/`, but `run_demos.sh` and
`start_demo_center.sh` hardcode `$HOME/demos` (`install.py` doesn't). Both scripts
source `ensure_demos_link.sh`, which creates the link on startup, so normally there is
nothing to do. To create it by hand:

```bash
ln -s ~/zgx-demo-center/demos ~/demos     # one-time
```

Re-create it if the checkout moves. `ensure_demos_link.sh` will refuse to repoint a link
that already resolves to a *different* checkout — that is deliberate, and it tells you
how to resolve the conflict. Fixing the scripts to derive their dir from `$0` would
remove the need, but diverges from the canonical HP scripts, and every contributed demo
hardcodes `~/demos` too.


---

## What's installed vs. what we skip

Running **headless**, we skip the on-device fullscreen "kiosk": Chromium,
`wmctrl`, the `start_demo_center.sh` wrapper, and the desktop icon. We use the
launcher web UI (**:8085**), the install API (**:8086**), and the per-demo Docker
stacks.

One-time heavy install (already done): miniforge/conda env `zgx`, Python deps,
Ollama models (`llama3.1:70b`, `nomic-embed-text`), native asset staging
(`~/yolov8s.onnx`, `~/zgx_videos`), TRT engine build (`~/yolov8s.trt`), and all
demo Docker images. Re-run with `bash demos/install.sh` (~10 min when warm).

> **Do not rename or move the `zgx` conda env.** Every native demo (Pipeline Pulse,
> AI Vision, Spectrum Duel) launches through the absolute path
> `~/miniforge3/envs/zgx/bin/python3`, as do the PII and pharmacy model downloaders.
> Renaming it takes all of them down at once. This happened on 2026-08-18 and cost a
> full day. First thing to check when native demos will not start:
>
> ```bash
> ls -l ~/miniforge3/envs/zgx/bin/python3     # must exist and be executable
> ```
>
> The start scripts now refuse to launch with a clear error when it is missing, and the
> launcher shows that error within seconds instead of spinning.

---

## Launch (on demand)

SSH into the device, then start the stack via the `~/demos` symlink (the script
expects that path):

```bash
cd ~/demos
bash run_demos.sh          # serves launcher :8085 + install API :8086
```

`run_demos.sh` runs in the foreground and prints the URLs; it kills anything
already on 8085/8086 first. To leave it running after you disconnect, start it
detached:

```bash
cd ~/demos
nohup bash run_demos.sh > ~/demo_center_run.log 2>&1 &
```

The individual demo containers are started/managed from the launcher UI
(Install / Launch buttons → the :8086 API drives `docker compose up`). They are
Docker containers, so they persist independently of `run_demos.sh`.

## Access

**Nothing needs forwarding.** Every server binds `0.0.0.0` and the launcher and
demo frontends derive their URLs from `window.location.hostname`, so all demo
and back-end ports ride whichever host you open the launcher on.

- **On-box** (monitor attached): `http://localhost:8085/zgx-demo-launcher.html`
- **Remote**: same path on the device's tailnet address — `run_demos.sh` prints
  the exact URL on startup. Use the raw tailnet IP if MagicDNS doesn't resolve.

> **This device is headless.** `start_demo_center.sh` (the fullscreen kiosk wrapper)
> launches Chromium, which dies with `Missing X server or $DISPLAY` and leaves only a
> "window did not map within 15s" warning in `logs/demo_center_launch.log`. Use
> `run_demos.sh` plus a browser on another machine. Tracked in `KNOWN-ISSUES.md`.

> **Do NOT deliver demos through VS Code Remote-SSH auto-forwarding.** It only
> tunnels ports it *detects*, so it routinely misses demo ports and the back-end
> ports `8083`/`8191`/`8192` that demo frontends call directly — one missed port
> is a blank iframe or a dead demo. This is the "ports don't forward reliably"
> symptom. Use VS Code for editing and the terminal only.

## Port map (authoritative — from launcher PORT_CONFLICTS / PORT_PATHS)

| Service      | Port         | Path                    | Notes                                  |
|--------------|--------------|-------------------------|----------------------------------------|
| Launcher UI  | 8085         | /zgx-demo-launcher.html |                                        |
| Install API  | 8086         | /health                 | FastAPI/uvicorn, SSE                   |
| pipeline     | 8088         | /                       | Pipeline Pulse                         |
| spectrum     | 8089         | /                       | TRT jam classifier + 70B briefs        |
| vision       | 8094         | /                       | YOLOv8s TRT                            |
| grant        | 8095         | /                       | Grant Machine — Ollama qwen3.6:35b     |
| searxng      | 8096         | /search                 | loopback only; optional, Discover web search |
| vehicle      | 8097         | /                       | Vehicle Inspection — Ollama mistral-small3.2 |
| maritime     | 8000         | /                       | shares 8000                            |
| pharmacy     | 8000         | /                       | shares 8000                            |
| pii          | 8000 (+8083) | /app/                   | shares 8000                            |
| ballistics   | 8093         | /                       | NAWSCL Aircraft Classification         |
| construction | 8091 (+8191) | /                       | back-end API on **8191**               |
| translation  | 8092 (+8192) | /                       | back-end API on **8192**; manual "Load Models" click in the demo |
| medical      | 8090         | /                       | **disabled** — needs Reachy robot      |

> maritime / pharmacy / pii share **:8000** — only one of the three runs at a
> time; the launcher stops one before starting another on that port.
>
> The parenthesised back-end ports (**8083**, **8191**, **8192**) are *not* in
> the launcher's `PORT_CONFLICTS` map. They resolve automatically when accessed
> as above; only a forwarding/proxy layer would need them enumerated.

## Stop / teardown

- Stop the launcher stack: `Ctrl+C` (foreground) or `kill` the `run_demos.sh`
  / `http.server` / `demo_install_server.py` PIDs (or `fuser -k 8085/tcp 8086/tcp`).
- Stop demo containers: `docker ps` then `docker stop <id>`, or `docker compose down`
  in the relevant `demos/<demo-dir>/`.
- The in-page **"Exit Demo Center"** button tears down the **entire** stack
  (servers + containers). Avoid clicking it remotely unless intended — you'd
  have to SSH back in and re-run `run_demos.sh`.

## Tailnet-only access

Every server binds `0.0.0.0`, so by default the launcher, the install API, each
demo, **and Ollama on :11434** are reachable from the LAN as well as the tailnet.
To restrict all of it to Tailscale:

```bash
sudo bash ~/demos/tools/lockdown-tailnet.sh --dry-run   # review every change first
sudo bash ~/demos/tools/lockdown-tailnet.sh             # apply
```

It keeps the services binding `0.0.0.0` on purpose and filters at the network
layer instead — the install API's readiness checks probe `127.0.0.1`, so a demo
bound only to the tailnet IP would read as "never started". It sets ufw to deny
inbound except loopback, `tailscale0`, and WireGuard's UDP 41641, and adds a
`DOCKER-USER` rule because **ufw does not filter Docker's published ports** —
Fed-Navy publishes `8000:8000` and would otherwise stay LAN-visible. That rule is
re-applied on boot by a small `zgx-docker-lan-block.service`, since dockerd
flushes the chain on every restart.

By default it leaves SSH open on the LAN as a lifeline; this box is headless, so
if Tailscale is down and LAN SSH is closed the only way in is physical access.
Pass `--no-lan-ssh` to close it anyway, `--pin-ollama` to bind Ollama to
localhost (nothing needs it on the wire — the Ollama-backed containers use
`network_mode: host` to reach it over loopback), and `--undo` to revert.

## Cautions

- Until `tools/lockdown-tailnet.sh` is run, servers bind `0.0.0.0` and are
  reachable on the LAN too, not just the tailnet. Ollama (`:11434`) is
  unauthenticated, and the install API (`:8086`) lets anyone who can reach it
  start, stop, or tear down demos.
- Nothing here is enabled at boot — after a reboot, SSH in and run `run_demos.sh`
  again. Demo containers will also be down until launched from the UI (unless
  Docker's restart policy brought them back).
