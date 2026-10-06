#!/usr/bin/env bash
# BSD-3-Clause; Copyright (c) 2026 Sichuan University contributors.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
: "${MODEL_ROOT:?set MODEL_ROOT to the already-provisioned read-only data_ckpt directory}"
: "${TECOOPS_API_ROOT:?set TECOOPS_API_ROOT to a local tecoops package parent with paired native MSDA}"
PYTHON=/home/py312/bin/python
test "$(readlink -f "$PYTHON")" = /usr/local/python/bin/python3.12
test -f "$TECOOPS_API_ROOT/tecoops/__init__.py"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /opt/tecoai/setvars.sh >/dev/null
export SDAA_ENABLE_COREDUMP_ON_EXCEPTION=0
export SDAA_VISIBLE_DEVICES="${SDAA_VISIBLE_DEVICES:-0}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$TECOOPS_API_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$PROJECT_DIR/data"
ensure_link() {
    local target=$1 link=$2
    test -e "$target"
    if test -e "$link" || test -L "$link"; then
        test "$(readlink -f "$link")" = "$(readlink -f "$target")"
    else
        ln -s -- "$(readlink -f "$target")" "$link"
    fi
}
ensure_link "$MODEL_ROOT/voc_coco" "$PROJECT_DIR/data/voc_coco"
ensure_link "$MODEL_ROOT/r50_deformable_detr_single_scale-checkpoint.pth" \
    "$PROJECT_DIR/r50_deformable_detr_single_scale-checkpoint.pth"
cd "$SCRIPT_DIR"
exec "$PYTHON" run_DeformableDETR.py "$@"
