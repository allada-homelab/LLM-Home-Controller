#!/usr/bin/env bash
set -euo pipefail

HA_CORE_DIR="/workspace/ha-core"
HA_CONFIG_DIR="/workspace/ha-config"
VENV_DIR="/home/vscode/.local/ha-venv"
INTEGRATION_SRC="/workspace/LLM-Home-Controller/custom_components/llm_home_controller"

echo ""
echo "==========================================="
echo "  LLM Home Controller: HA Core Dev Setup"
echo "==========================================="
echo ""

# --- Step 0: Fix ownership of Docker named volumes ---
# Named volumes and parent dirs may be created as root; ensure vscode user owns them.
sudo chown -R "$(id -u):$(id -g)" "$HA_CORE_DIR" "$HA_CONFIG_DIR" "$VENV_DIR" /home/vscode/.local 2>/dev/null || true

# --- Step 1: Clone HA Core ---
if [ ! -d "$HA_CORE_DIR/.git" ]; then
    echo "[1/5] Cloning Home Assistant Core..."
    git clone --depth 1 https://github.com/home-assistant/core.git "$HA_CORE_DIR"
else
    echo "[1/5] Home Assistant Core already cloned, pulling latest..."
    cd "$HA_CORE_DIR" && git pull --ff-only 2>/dev/null || echo "     (pull skipped — detached HEAD or no remote changes)"
fi

# --- Step 2: Detect Python version and create venv ---
PYTHON_VERSION="3.13"
if [ -f "$HA_CORE_DIR/.python-version" ]; then
    PYTHON_VERSION=$(tr -d '[:space:]' < "$HA_CORE_DIR/.python-version")
fi
echo "[2/5] Setting up Python ${PYTHON_VERSION} virtual environment..."
uv python install "$PYTHON_VERSION"
if [ ! -d "$VENV_DIR/bin" ]; then
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi
export PATH="$VENV_DIR/bin:$PATH"

# --- Step 3: Install HA Core ---
echo "[3/5] Installing Home Assistant Core + test dependencies..."
echo "     (this may take several minutes on first run)"
cd "$HA_CORE_DIR"
uv pip install \
    -e "." \
    -r requirements_test.txt \
    colorlog \
    --config-settings editable_mode=compat 2>&1 | tail -5

# --- Step 4: Install our integration dependencies ---
echo "[4/5] Installing LLM Home Controller dependencies..."
cd /workspace/LLM-Home-Controller
# Only install dependencies, NOT the package itself as an editable install.
# The integration is loaded via symlink (step 5), and an editable install
# adds a sys.path hook that conflicts with HA's custom_components loader.
if [ -f "requirements.txt" ]; then
    uv pip install -r requirements.txt 2>&1 | tail -3
else
    echo "     (no requirements.txt — skipping)"
fi

# --- Step 5: Configure HA ---
echo "[5/5] Configuring Home Assistant..."
mkdir -p "$HA_CONFIG_DIR/custom_components"
ln -sfn "$INTEGRATION_SRC" "$HA_CONFIG_DIR/custom_components/llm_home_controller"

if [ ! -f "$HA_CONFIG_DIR/configuration.yaml" ]; then
    hass --script ensure_config -c "$HA_CONFIG_DIR" 2>/dev/null || true
fi

# Enable debug logging for our integration
if [ -f "$HA_CONFIG_DIR/configuration.yaml" ] && ! grep -q "llm_home_controller" "$HA_CONFIG_DIR/configuration.yaml" 2>/dev/null; then
    cat >> "$HA_CONFIG_DIR/configuration.yaml" << 'YAML'

logger:
  default: info
  logs:
    custom_components.llm_home_controller: debug
YAML
fi

# --- Step 6: Pull default Ollama model ---
echo "[6/6] Pulling default Ollama model (qwen3:4b)..."
for i in 1 2 3 4 5; do
    if curl -sf http://ollama:11434/api/tags >/dev/null 2>&1; then
        curl -s http://ollama:11434/api/pull -d '{"name":"qwen3:4b"}' | while IFS= read -r line; do
            status=$(echo "$line" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)
            [ -n "$status" ] && printf "\r     %s" "$status"
        done
        echo ""
        break
    fi
    echo "     Waiting for Ollama to be ready... ($i/5)"
    sleep 3
done

echo ""
echo "==========================================="
echo "  Setup Complete!"
echo "==========================================="
echo ""
echo "  Start HA:     hass -c /workspace/ha-config"
echo "  Or press F5 in VS Code to launch with debugger"
echo ""
echo "  HA UI:         http://localhost:8123"
echo "  Ollama API:    http://ollama:11434  (from inside container)"
echo "                 http://localhost:11434 (from host)"
echo ""
echo "  Update HA:     cd /workspace/ha-core && git pull && uv pip install -e ."
echo "  Reset config:  rm -rf /workspace/ha-config && bash scripts/setup-ha-core.sh"
echo ""
echo "  Your custom component is symlinked — code changes"
echo "  take effect after restarting HA (no rebuild needed)."
echo ""
