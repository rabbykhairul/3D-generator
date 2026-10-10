#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$project_root/artifacts/triposr-src"
weights_dir="$project_root/artifacts/triposr-weights"
venv_dir="$project_root/artifacts/triposr-venv"

if [[ ! -f "$source_dir/run.py" ]]; then
  git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git "$source_dir"
fi
if [[ ! -x "$venv_dir/bin/python" ]]; then
  if [[ -x "$project_root/artifacts/segmenter-venv/bin/python" ]]; then
    "$project_root/artifacts/segmenter-venv/bin/python" -m venv "$venv_dir"
  else
    python3 -m venv "$venv_dir"
  fi
fi
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install torch torchvision
"$venv_dir/bin/python" -m pip install -r "$source_dir/requirements.txt"

mkdir -p "$weights_dir"
if [[ ! -f "$weights_dir/model.ckpt" || ! -f "$weights_dir/config.yaml" ]]; then
  if [[ -x "$project_root/artifacts/sam3-venv/bin/hf" ]]; then
    "$project_root/artifacts/sam3-venv/bin/hf" download stabilityai/TripoSR \
      model.ckpt config.yaml --local-dir "$weights_dir"
  else
    "$venv_dir/bin/python" -m pip install 'huggingface_hub[cli]'
    "$venv_dir/bin/hf" download stabilityai/TripoSR \
      model.ckpt config.yaml --local-dir "$weights_dir"
  fi
fi

"$venv_dir/bin/python" "$project_root/scripts/run_triposr_probe.py"
