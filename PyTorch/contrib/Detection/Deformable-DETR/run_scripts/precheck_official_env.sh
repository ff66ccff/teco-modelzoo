#!/usr/bin/env bash
# BSD-3-Clause; Copyright (c) 2026 Sichuan University contributors.
#
# Static precheck for the official ModelZoo environment (Python 3.11 / PyTorch 2.7.1).
#
# This script only imports and hashes.  It never installs, upgrades or removes a
# package, never writes into the source tree and never launches an SDAA kernel;
# it exists so that a fresh official session can prove its stack *before* the GPU
# window is spent.  Every item prints OK / FAIL / SKIP and the exit code is
# non-zero when a required item fails.
#
# Usage:
#   PYTHON=/path/to/python3.11 \
#   MODEL_ROOT=/path/to/provisioned/data_ckpt \
#   TECOOPS_API_ROOT=/path/to/project-local/native/api \
#   bash run_scripts/precheck_official_env.sh
#
# Overrides: EXPECTED_PYTHON_MINOR (default 3.11), EXPECTED_TORCH_PREFIX (default 2.7.1).
# The vendor py3.12 stack can be inspected with EXPECTED_PYTHON_MINOR=3.12
# EXPECTED_TORCH_PREFIX=2.12.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-/home/py312/bin/python}"
EXPECTED_PYTHON_MINOR="${EXPECTED_PYTHON_MINOR:-3.11}"
EXPECTED_TORCH_PREFIX="${EXPECTED_TORCH_PREFIX:-2.7.1}"
SETVARS="${TECOAI_SETVARS:-/opt/tecoai/setvars.sh}"

failures=0
ok() { printf 'PRECHECK: OK   %s\n' "$*"; }
bad() { printf 'PRECHECK: FAIL %s\n' "$*" >&2; failures=$((failures + 1)); }
skip() { printf 'PRECHECK: SKIP %s\n' "$*"; }

printf 'PRECHECK: Deformable-DETR official-environment precheck (static; no install, no GPU work)\n'

# --- 1. interpreter ---------------------------------------------------------
if test -x "$PYTHON"; then
    resolved="$(readlink -f "$PYTHON" 2>/dev/null || printf '%s' "$PYTHON")"
    version="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if test "$version" = "$EXPECTED_PYTHON_MINOR"; then
        ok "interpreter $PYTHON -> $resolved is Python $version"
    else
        bad "interpreter $PYTHON -> $resolved reports Python ${version:-unknown}, expected $EXPECTED_PYTHON_MINOR (official ModelZoo). Set EXPECTED_PYTHON_MINOR=3.12 to inspect the vendor stack, or PYTHON=/path/to/python3.11"
    fi
else
    bad "interpreter '$PYTHON' is not executable; set PYTHON to the official Python 3.11 path or to /home/py312/bin/python"
    printf 'PRECHECK: summary %d failure(s); aborting early\n' "$failures" >&2
    exit 1
fi

# --- 2. SDAA runtime environment (process-local only) -----------------------
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if test -f "$SETVARS"; then
    set +u
    # shellcheck disable=SC1090
    . "$SETVARS" >/dev/null
    set -u
    ok "sourced $SETVARS (process-local environment only)"
else
    skip "$SETVARS not found; assuming the SDAA runtime is already exported"
fi

# --- 3. paired tecoops binding ---------------------------------------------
# The supported dependency is the INSTALLED tecoops wheel
# (pip install tecoops-*.whl).  TECOOPS_API_ROOT is an optional source-tree
# development override; when it is set it must contain the paired package.
if test -n "${TECOOPS_API_ROOT:-}"; then
    if test -f "$TECOOPS_API_ROOT/tecoops/__init__.py"; then
        export PYTHONPATH="$TECOOPS_API_ROOT${PYTHONPATH:+:$PYTHONPATH}"
        ok "TECOOPS_API_ROOT=$TECOOPS_API_ROOT (development override; prepended to PYTHONPATH for the probes below)"
    else
        bad "TECOOPS_API_ROOT=$TECOOPS_API_ROOT contains no tecoops/__init__.py"
    fi
else
    ok "TECOOPS_API_ROOT unset -> using the installed tecoops wheel (pip install tecoops-*.whl)"
fi

# --- 4. real stack capabilities (torch / torchvision / SDAA / tecoops) ------
if "$PYTHON" "$PROJECT_DIR/runtime_contract.py" --check \
    --role "run_scripts/precheck_official_env.sh" \
    --tecoops-root "${TECOOPS_API_ROOT:-}"; then
    ok "runtime capability contract passed (torch, torchvision, SDAA device, paired tecoops)"
else
    bad "runtime capability contract failed; see the [runtime_contract] lines above"
fi

# --- 5. torch version gate --------------------------------------------------
# The vendor/official torch import prints a TECO banner on stdout, so every probe
# below is marker-prefixed and filtered instead of trusting raw stdout.
torch_version="$("$PYTHON" -c 'import torch; print("TORCH_VERSION=" + torch.__version__)' 2>/dev/null \
    | sed -n 's/^TORCH_VERSION=//p' | tail -n 1 || true)"
case "$torch_version" in
    "$EXPECTED_TORCH_PREFIX"*)
        ok "torch $torch_version matches the expected prefix $EXPECTED_TORCH_PREFIX"
        ;;
    "")
        bad "cannot query torch.__version__ with $PYTHON"
        ;;
    *)
        bad "torch $torch_version does not match the expected prefix $EXPECTED_TORCH_PREFIX; do not pip-install torch over the official/vendor build"
        ;;
esac

# --- 6. tecoops resolution self-evidence ------------------------------------
tecoops_file="$("$PYTHON" -c 'import tecoops; print("TECOOPS_FILE=" + tecoops.__file__)' 2>/dev/null \
    | sed -n 's/^TECOOPS_FILE=//p' | tail -n 1 || true)"
if test -z "$tecoops_file"; then
    bad "cannot import tecoops with $PYTHON (install the paired wheel: pip install tecoops-*.whl)"
elif test -n "${TECOOPS_API_ROOT:-}"; then
    case "$tecoops_file" in
        "$TECOOPS_API_ROOT"/*) ok "tecoops.__file__=$tecoops_file (resolved under TECOOPS_API_ROOT)" ;;
        *) bad "tecoops resolved to $tecoops_file, which is not under TECOOPS_API_ROOT=$TECOOPS_API_ROOT (shadowing)" ;;
    esac
else
    ok "tecoops.__file__=$tecoops_file (installed wheel)"
fi

# --- 7. read-only assets ----------------------------------------------------
check_asset() {
    local path=$1 expected=$2 label=$3
    if test ! -f "$path"; then
        bad "$label missing: $path"
        return
    fi
    local actual
    actual="$(sha256sum "$path" | cut -d' ' -f1)"
    if test "$actual" = "$expected"; then
        ok "$label sha256 matches ($expected)"
    else
        bad "$label sha256 mismatch: got $actual, expected $expected"
    fi
}
if test -n "${MODEL_ROOT:-}"; then
    check_asset "$MODEL_ROOT/voc_coco/train/train.json" \
        e511d3a0e39f45162887e4d9f0bed63bf8f01e4c421d965445185b72990f1ab5 "VOC2007 trainval JSON"
    check_asset "$MODEL_ROOT/voc_coco/val/val.json" \
        14edfb83a2525773c9e5b6e6d4bfa57a866de1cb3ca164c46afe0ac97629b561 "VOC2007 test JSON"
    check_asset "$MODEL_ROOT/r50_deformable_detr_single_scale-checkpoint.pth" \
        d442fb2365d6e9640347b2b38686089d13cd55a5c3790d63e3602d079791fed1 "official pretrained checkpoint"
else
    bad "MODEL_ROOT is unset; export the provisioned read-only data_ckpt directory to verify the three pinned assets"
fi

# --- summary ----------------------------------------------------------------
if test "$failures" -eq 0; then
    printf 'PRECHECK: summary all checks passed; the environment matches the documented contract\n'
    exit 0
fi
printf 'PRECHECK: summary %d failure(s); fix the items above before starting the GPU window\n' "$failures" >&2
exit 1
