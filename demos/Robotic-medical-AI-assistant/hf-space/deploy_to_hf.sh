#!/bin/bash
# ===========================================================================
# deploy_to_hf.sh
# ===========================================================================
# Uploads the consent agent app to HuggingFace Spaces,
# then reinstalls it on the Reachy Mini.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login   (with a write-access token)
#
# Usage:
#   chmod +x deploy_to_hf.sh
#   ./deploy_to_hf.sh
#
# Optional env vars:
#   HF_REPO_ID    — default: curtburk/consent-agent-reachy
#   REACHY_HOST   — default: reachy-mini.local
# ===========================================================================

set -euo pipefail

HF_REPO_ID="${HF_REPO_ID:-curtburk/consent-agent-reachy}"
REACHY_HOST="${REACHY_HOST:-reachy-mini.local}"
REACHY_API="http://${REACHY_HOST}:8000/api"

# Path to the HF Space files (this script lives in the hf-space dir)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPACE_DIR="${SCRIPT_DIR}"

echo "============================================"
echo " Deploying Consent Agent to HuggingFace"
echo " Repo: ${HF_REPO_ID}"
echo " Robot: ${REACHY_HOST}"
echo "============================================"
echo ""

# ---- Upload all files to HF Space ----
echo "[1/3] Uploading files to HuggingFace Space..."

python3 << PYEOF
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "${HF_REPO_ID}"
space_dir = "${SPACE_DIR}"

# Ensure the space exists (creates if not)
try:
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="static", exist_ok=True)
    print(f"  Space '{repo_id}' ready.")
except Exception as e:
    print(f"  Note: {e}")

# Files to upload (path_on_disk -> path_in_repo)
files = [
    # Root-level Space files
    ("README.md", "README.md"),
    ("index.html", "index.html"),
    ("style.css", "style.css"),
    ("pyproject.toml", "pyproject.toml"),
    # Python package
    ("consent_agent_reachy/__init__.py", "consent_agent_reachy/__init__.py"),
    ("consent_agent_reachy/main.py", "consent_agent_reachy/main.py"),
]

for local_rel, repo_path in files:
    local_path = os.path.join(space_dir, local_rel)
    if not os.path.exists(local_path):
        print(f"  SKIP (not found): {local_path}")
        continue
    print(f"  Uploading: {repo_path}")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=repo_path,
        repo_id=repo_id,
        repo_type="space",
    )

print("")
print("✅ All files uploaded to HuggingFace Space!")
print(f"   https://huggingface.co/spaces/{repo_id}")
PYEOF

echo ""

# ---- Remove old version from Reachy ----
echo "[2/3] Removing old version from Reachy..."

curl -sf -X POST "${REACHY_API}/apps/remove/consent_agent_reachy" \
    && echo "  Old version removed." \
    || echo "  No existing version found (OK)."

echo "  Waiting for cleanup..."
sleep 5

# ---- Install new version on Reachy ----
echo "[3/3] Installing new version on Reachy..."

curl -sf -X POST "${REACHY_API}/apps/install" \
    -H "Content-Type: application/json" \
    -d "{
        \"name\": \"consent_agent_reachy\",
        \"source_kind\": \"hf_space\",
        \"description\": \"Medical triage voice agent (local Whisper + vLLM + Piper)\",
        \"url\": \"https://huggingface.co/spaces/${HF_REPO_ID}\"
    }" \
    && echo "  ✅ Installed on Reachy!" \
    || echo "  ❌ Install failed — check if Reachy is reachable at ${REACHY_HOST}"

# ---- Direct push to robot (bypasses pip caching issues) ----
echo ""
echo "[3b/3] Pushing main.py directly to robot..."
scp "${SPACE_DIR}/consent_agent_reachy/main.py" \
    "pollen@${REACHY_HOST}:/venvs/apps_venv/lib/python3.12/site-packages/consent_agent_reachy/main.py" \
    && echo "  ✅ main.py pushed to robot." \
    || echo "  ⚠️  SCP failed — you may need to push manually."

ssh "pollen@${REACHY_HOST}" "rm -rf /venvs/apps_venv/lib/python3.12/site-packages/consent_agent_reachy/__pycache__" 2>/dev/null

echo ""
echo "============================================"
echo " Deployment complete!"
echo ""
echo " HF Space: https://huggingface.co/spaces/${HF_REPO_ID}"
echo " Robot:    http://${REACHY_HOST}:8000"
echo ""
echo " Make sure the ZGX AI API container is running"
echo " before starting the app on the robot."
echo "============================================"