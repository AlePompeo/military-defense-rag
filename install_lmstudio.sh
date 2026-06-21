#!/usr/bin/env bash
# LM Studio installer for Fedora 43 (x86_64)
# Installs the lms CLI, configures PATH, and optionally downloads a model.
set -euo pipefail

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()      { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── sanity checks ─────────────────────────────────────────────────────────────
[[ $(uname -m) == "x86_64" ]] || die "Only x86_64 is supported by this script."
[[ -f /etc/fedora-release ]]   || warn "This script targets Fedora; continuing anyway."

# ── 1. system dependencies ────────────────────────────────────────────────────
info "Installing system dependencies (fuse, fuse-libs)..."
if ! rpm -q fuse fuse-libs &>/dev/null; then
    sudo dnf install -y fuse fuse-libs
    ok "FUSE installed."
else
    ok "FUSE already present."
fi

# ldconfig is required by the lms installer
if ! command -v ldconfig &>/dev/null; then
    info "Installing ldconfig (glibc-devel)..."
    sudo dnf install -y glibc-devel
fi

# ── 2. install lms CLI ────────────────────────────────────────────────────────
LMS_BIN="$HOME/.lmstudio/bin/lms"

if [[ -x "$LMS_BIN" ]]; then
    INSTALLED_VER=$("$LMS_BIN" --version 2>/dev/null || echo "unknown")
    ok "lms CLI already installed (version: $INSTALLED_VER). Skipping download."
else
    info "Downloading and installing lms CLI from lmstudio.ai..."
    curl -fsSL https://lmstudio.ai/install.sh | sh
    ok "lms CLI installed."
fi

# ── 3. configure PATH ─────────────────────────────────────────────────────────
LMS_BIN_DIR="$HOME/.lmstudio/bin"
SHELL_RC=""

case "$SHELL" in
    */bash) SHELL_RC="$HOME/.bashrc" ;;
    */zsh)  SHELL_RC="$HOME/.zshrc"  ;;
    */fish) SHELL_RC="$HOME/.config/fish/config.fish" ;;
    *)      SHELL_RC="$HOME/.profile" ;;
esac

PATH_LINE="export PATH=\"\$PATH:$LMS_BIN_DIR\""

if [[ -f "$SHELL_RC" ]] && grep -qF "$LMS_BIN_DIR" "$SHELL_RC"; then
    ok "PATH already configured in $SHELL_RC."
else
    echo "" >> "$SHELL_RC"
    echo "# lm studio cli" >> "$SHELL_RC"
    echo "$PATH_LINE" >> "$SHELL_RC"
    ok "Added $LMS_BIN_DIR to PATH in $SHELL_RC."
fi

export PATH="$PATH:$LMS_BIN_DIR"

# ── 4. verify installation ────────────────────────────────────────────────────
info "Verifying lms CLI..."
if ! command -v lms &>/dev/null && [[ ! -x "$LMS_BIN" ]]; then
    die "lms not found after install. Check the output above for errors."
fi

LMS="$LMS_BIN"
LMS_VER=$("$LMS" --version 2>/dev/null || echo "unknown")
ok "lms version: $LMS_VER"

# ── 5. optional: download a model ─────────────────────────────────────────────
echo ""
echo -e "${BLUE}Available recommended models for this RAG:${NC}"
echo "  1) lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF  (fast, good quality)"
echo "  2) lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF    (lighter)"
echo "  3) Skip (configure manually in LM Studio later)"
echo ""
read -rp "Choose a model to download [1/2/3]: " MODEL_CHOICE

case "$MODEL_CHOICE" in
    1)
        MODEL_ID="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"
        info "Downloading $MODEL_ID (this may take several minutes)..."
        "$LMS" get "$MODEL_ID"
        ok "Model downloaded."
        ;;
    2)
        MODEL_ID="lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF"
        info "Downloading $MODEL_ID..."
        "$LMS" get "$MODEL_ID"
        ok "Model downloaded."
        ;;
    *)
        warn "Skipped model download. Use 'lms get <model-id>' later."
        ;;
esac

# ── 6. start server ───────────────────────────────────────────────────────────
echo ""
read -rp "Start the LM Studio server now on port 1234? [y/N]: " START_SERVER

if [[ "${START_SERVER,,}" == "y" ]]; then
    info "Starting LM Studio server..."
    "$LMS" server start --port 1234 &
    SERVER_PID=$!
    sleep 2

    if curl -sf http://localhost:1234/v1/models >/dev/null 2>&1; then
        ok "Server is running at http://localhost:1234/v1 (PID $SERVER_PID)"
        ok "Run your RAG: python main.py query -q \"your question\""
    else
        warn "Server started (PID $SERVER_PID) but not yet responding — may still be loading the model."
        warn "Check with: curl http://localhost:1234/v1/models"
    fi
else
    echo ""
    info "To start the server manually later:"
    echo "    lms server start --port 1234"
    echo ""
    info "To run the RAG:"
    echo "    python main.py query -q \"your question\""
fi

# ── 7. summary ────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN} LM Studio setup complete${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  CLI binary : $LMS_BIN"
echo "  Models dir : $HOME/.lmstudio/models"
echo "  Server URL : http://localhost:1234/v1"
echo ""
echo "  Useful commands:"
echo "    lms server start          # start server"
echo "    lms server stop           # stop server"
echo "    lms ps                    # list running models"
echo "    lms ls                    # list downloaded models"
echo "    lms get <model-id>        # download a model"
echo ""
warn "Reload your shell to pick up PATH changes: source $SHELL_RC"
