#!/usr/bin/env bash
set -euo pipefail

python -m eyewear_vto.migrate
python -m eyewear_vto.preflight
exec "$@"
