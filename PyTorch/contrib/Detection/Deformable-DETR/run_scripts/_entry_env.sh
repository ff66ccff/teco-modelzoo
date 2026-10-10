#!/usr/bin/env bash
# BSD-3-Clause; Copyright (c) 2026 Sichuan University contributors.
# shellcheck shell=bash
# Shared environment and capability binding for the Deformable-DETR delivery entries.
#
# Sourced by run_scripts/*.sh.  It selects the interpreter (PYTHON overrides the
# vendor default), sources the SDAA runtime, pins the paired tecoops package in
# front of PYTHONPATH (no silent site-packages shadowing) and then runs the
# capability contract in ../runtime_contract.py.  Nothing here installs,
# upgrades or modifies the environment.

detr_fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

detr_ensure_link() {
    local target=$1 link=$2
    test -e "$target" || detr_fail "required read-only asset is missing: $target (check MODEL_ROOT)"
    if test -e "$link" || test -L "$link"; then
        test "$(readlink -f "$link")" = "$(readlink -f "$target")" ||
            detr_fail "existing path $link does not point at $target"
    else
        ln -s -- "$(readlink -f "$target")" "$link"
    fi
}

detr_stage_assets() {
    local project_dir=$1
    mkdir -p "$project_dir/data"
    detr_ensure_link "$MODEL_ROOT/voc_coco" "$project_dir/data/voc_coco"
    detr_ensure_link "$MODEL_ROOT/r50_deformable_detr_single_scale-checkpoint.pth" \
        "$project_dir/r50_deformable_detr_single_scale-checkpoint.pth"
}

detr_setup_env() {
    local project_dir=$1 role=$2
    local setvars="${TECOAI_SETVARS:-/opt/tecoai/setvars.sh}"

    PYTHON="${PYTHON:-/home/py312/bin/python}"
    export PYTHON
    test -x "$PYTHON" ||
        detr_fail "interpreter '$PYTHON' is not executable; set PYTHON=/home/py312/bin/python (vendor, Python 3.12) or the official ModelZoo Python 3.11 path"

    : "${MODEL_ROOT:?set MODEL_ROOT to the already-provisioned read-only data_ckpt directory}"
    # The supported dependency is the INSTALLED tecoops wheel
    # (pip install tecoops-*.whl).  TECOOPS_API_ROOT is an optional development
    # override for a source-tree package; when set it is validated and prepended
    # to PYTHONPATH, otherwise site-packages is used as-is.
    if test -n "${TECOOPS_API_ROOT:-}"; then
        test -f "$TECOOPS_API_ROOT/tecoops/__init__.py" ||
            detr_fail "TECOOPS_API_ROOT=$TECOOPS_API_ROOT contains no tecoops/__init__.py; point it at the parent of the paired Teco-Ops PR #40 package, or unset it to use the installed wheel"
        printf 'INFO: TECOOPS_API_ROOT override=%s (development form; the installed wheel is the default)\n' "$TECOOPS_API_ROOT"
    fi

    printf 'INFO: entry interpreter PYTHON=%s -> %s\n' "$PYTHON" \
        "$(readlink -f "$PYTHON" 2>/dev/null || printf '%s' "$PYTHON")"

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
    if test -f "$setvars"; then
        set +u
        # shellcheck disable=SC1090
        . "$setvars" >/dev/null
        set -u
    else
        printf 'WARN: %s not found; the SDAA runtime must already be present in the environment\n' "$setvars" >&2
    fi
    export SDAA_ENABLE_COREDUMP_ON_EXCEPTION=0
    export SDAA_VISIBLE_DEVICES="${SDAA_VISIBLE_DEVICES:-0}"
    export PYTHONDONTWRITEBYTECODE=1
    if test -n "${TECOOPS_API_ROOT:-}"; then
        export PYTHONPATH="$TECOOPS_API_ROOT${PYTHONPATH:+:$PYTHONPATH}"
    fi

    "$PYTHON" "$project_dir/runtime_contract.py" --check --role "$role" \
        --tecoops-root "${TECOOPS_API_ROOT:-}" ||
        detr_fail "runtime capability contract failed for $role (see the [runtime_contract] lines above)"
}
