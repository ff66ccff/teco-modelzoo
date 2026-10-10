#!/usr/bin/env bash
# BSD-3-Clause; Copyright (c) 2026 Sichuan University contributors.
# Reference correctness entry: native MSDA forward/gradient oracle at real VOC
# model boundaries (see ../verify_native_voc.py).  Uses the same interpreter,
# SDAA runtime and paired tecoops binding as run_scripts/test.sh, so the oracle
# can never silently resolve an unrelated site-packages tecoops wheel.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
# shellcheck source=_entry_env.sh
. "$SCRIPT_DIR/_entry_env.sh"
detr_setup_env "$PROJECT_DIR" "run_scripts/verify_native_voc.sh"
detr_stage_assets "$PROJECT_DIR"
cd "$PROJECT_DIR"
exec "$PYTHON" verify_native_voc.py "$@"
