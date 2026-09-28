#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "======================================================"
echo "🚀 INICIANDO ESTEIRA AUTOMÁTICA - MONEYPRINTERTURBO"
echo "======================================================"

# 1. Verifica/Inicia o ComfyUI Server (porta 8188)
if ! lsof -i :8188 >/dev/null 2>&1; then
    echo "⚙️  Iniciando ComfyUI Server em segundo plano..."
    /Users/leandrobocon/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python /Users/leandrobocon/ComfyUI-Installs/ComfyUI/ComfyUI/main.py --listen 127.0.0.1 --port 8188 > /dev/null 2>&1 &
    sleep 3
else
    echo "✅ ComfyUI Server já está ativo (porta 8188)."
fi

# 2. Verifica/Inicia a Bridge de Automação (porta 8000)
if ! lsof -i :8000 >/dev/null 2>&1; then
    echo "⚙️  Iniciando Bridge do ComfyUI (Qwen Image 2.1 / FLUX) em segundo plano..."
    /Users/leandrobocon/ComfyUI-Installs/ComfyUI/ComfyUI/.venv/bin/python "$DIR/scripts/comfy_openai_bridge.py" > /dev/null 2>&1 &
    sleep 1
else
    echo "✅ Bridge do ComfyUI já está ativa (porta 8000)."
fi

# 3. Inicia o MoneyPrinterTurbo WebUI
echo "🎬 Abrindo MoneyPrinterTurbo..."
./webui.sh
