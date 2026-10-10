#!/usr/bin/env bash
# BSD-3-Clause; Copyright (c) 2026 Sichuan University contributors.
# Delivery entry: bounded VOC2007 single-scale SDAA training / evaluation / resume.
#
# The interpreter is selected by PYTHON (default: the vendor
# /home/py312/bin/python) and validated by the capability contract in
# ../runtime_contract.py -- Python 3.11 (official ModelZoo) and 3.12 (vendor),
# torch/torchvision, a usable SDAA device and the paired tecoops package are all
# required.  CPU fallback is never accepted.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=_entry_env.sh
. "$SCRIPT_DIR/_entry_env.sh"
detr_setup_env "$PROJECT_DIR" "run_scripts/test.sh"
detr_stage_assets "$PROJECT_DIR"
cd "$SCRIPT_DIR"
exec "$PYTHON" run_DeformableDETR.py "$@"
