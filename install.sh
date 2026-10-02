#!/usr/bin/env bash
# Make the File Toolkit runnable from anywhere by symlinking the
# 'file-toolkit' launcher into ~/.local/bin.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${HOME}/.local/bin"

chmod +x "$SCRIPT_DIR/file-toolkit"
mkdir -p "$BIN_DIR"
ln -sf "$SCRIPT_DIR/file-toolkit" "$BIN_DIR/file-toolkit"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo "Note: $BIN_DIR is not in your PATH yet."
        echo "Add this line to your shell rc file (~/.bashrc or ~/.zshrc):"
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        ;;
esac

echo "Installed: run the toolkit from anywhere with 'file-toolkit'."
