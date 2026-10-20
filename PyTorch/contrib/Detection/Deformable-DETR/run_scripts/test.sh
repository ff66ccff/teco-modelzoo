#!/usr/bin/env bash
# BSD-3-Clause; Copyright (c) 2026 Sichuan University contributors.
# Delivery entry: bounded VOC2007 single-scale SDAA training / evaluation / resume.
#
# The interpreter is selected by PYTHON (default: the vendor
# /home/py312/bin/python) and validated by the capability contract in
# ../runtime_contract.py -- Python 3.11 (official ModelZoo) and 3.12 (vendor),
# torch/torchvision, a usable SDAA device and the paired tecoops package are all
# required.  CPU fallback is never accepted.
#
# Parameterized ModelZoo-style CLI.  `bash test.sh --help` prints usage and
# exits before any environment variable is consumed, so it also works on a bare
# login shell with neither MODEL_ROOT nor PYTHON set.  Alias normalization
# (--bs/--batch_size -> --batchsize, --epochs -> --epoch, --output_dir ->
# --output-dir, --dataset_path/--model_root -> MODEL_ROOT), the PYTHON and
# MODEL_ROOT defaults and the fail-closed read-only asset checks all happen
# HERE, before run_scripts/_entry_env.sh is sourced.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEFAULT_MODEL_ROOT=/root/competition-assets/deformable-detr-voc2007-20260717/payload/data_ckpt
DEFAULT_PYTHON=/home/py312/bin/python
CKPT=r50_deformable_detr_single_scale-checkpoint.pth

usage() {
    cat <<'USAGE'
Deformable-DETR SDAA delivery entry (run_scripts/test.sh)

Usage: bash test.sh [options]

Options:
  --output-dir DIR        output directory, a NEW empty directory required for
                          each invocation
                          [default: <project>/outputs/run-<UTC timestamp>]
  --output_dir DIR        alias of --output-dir
  --bs N                  alias of --batchsize N                     [default: 2]
  --batch_size N          alias of --batchsize N
  --batchsize N           minibatch size (aggregate card view)       [default: 2]
  --epochs N              alias of --epoch N; the validated official R50
  --epoch N               single-scale recipe is FIXED at 20 epochs, so any
                          other value is rejected up front -- use
                          --max-train-steps/--max-seconds for bounded runs
                                                                     [default: 20]
  --device D              'sdaa' or 'sdaa:0' (single card; no CPU fallback)
                                                                     [default: sdaa:0]
  --nproc_per_node N      must be 1 (single-process single-card entry)
  --model_name NAME       must be DeformableDETR
  --dataset_path DIR      read-only asset root; a data_ckpt directory or its
  --model_root DIR        parent is accepted and must contain voc_coco/ and
                          r50_deformable_detr_single_scale-checkpoint.pth
                          (fail-closed check)
                          [default: /root/competition-assets/
                            deformable-detr-voc2007-20260717/payload/data_ckpt]
  --seed N                training seed                            [default: 42]
  --resume PATH           resume from a checkpoint produced by this entry
  --max-train-steps N     stop after N completed minibatches    [default: 0 = unbounded]
  --max-seconds S         stop after a completed minibatch once S wall seconds
                          are spent                             [default: 0 = unbounded]
  --eval-images N         evaluate the first N test images [default: 0 = all 4952]
  --checkpoint-every N    save a checkpoint every N steps         [default: 1000]
  --eval                  evaluation-only run (requires --resume)
  --audit-msda            count native MSDA fwd/bwd calls (default behaviour)
  --no-audit-msda         disable the MSDA call audit
  --lr LR                 official optimizer overrides supported natively by
  --lr_backbone LR        train_sdaa.py; forwarded ONLY when given, so the
  --weight_decay WD       default keeps the official recipe
  --clip_max_norm N       lr=2e-5 / lr_backbone=2e-6
  --coco_path DIR         dataset directory (default: ./data/voc_coco via
                          MODEL_ROOT/voc_coco)
  -h, --help              print this message and exit

Environment (all optional):
  PYTHON                  interpreter [default: /home/py312/bin/python]
  MODEL_ROOT              read-only data_ckpt root (see --dataset_path)

Examples:
  cd run_scripts && bash test.sh --help
  cd run_scripts && bash test.sh --max-train-steps 1 --eval-images 2
  cd run_scripts && bash test.sh --bs 2 --device sdaa --max-train-steps 1 --eval-images 2
  bash run_scripts/test.sh --eval --resume /tmp/detr-run/checkpoint.pth
USAGE
}

entry_fail() {
    printf 'ERROR: %s\n' "$*" >&2
    printf 'hint: run "bash test.sh --help" for the accepted options\n' >&2
    exit 2
}

entry_uint() { case $1 in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac; }
entry_int()  { case $1 in -|''|-[!0-9]*|*[!0-9-]*) return 1 ;; *) return 0 ;; esac; }
entry_float() {
    # Accepts optional sign, plain/exponent decimal notation (e.g. 2e-5).
    # Valid: 2e-5, 2E-6, .5, 5., 1.5, -2e-5, +1.5.  Invalid: '', '.', '1e', 'e5'.
    local NUM_RE
    local form
    NUM_RE='^[+-]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
    form=$1
    case $form in '') return 1 ;; esac
    if [[ $form =~ $NUM_RE ]]; then
        return 0
    fi
    return 1
}

value_flag() {
    case $1 in
        --output-dir|--output_dir|\
--model_name|--nproc_per_node|\
--bs|--batch_size|--batchsize|\
--epochs|--epoch|\
--device|\
--dataset_path|--dataset-path|--model_root|--model-root|\
--resume|--seed|\
--max-train-steps|--max-seconds|--eval-images|--checkpoint-every|\
--lr|--lr_backbone|--weight_decay|--clip_max_norm|--coco_path|--coco-path) return 0 ;;
        *) return 1 ;;
    esac
}

OUTPUT_DIR='' MODEL_NAME='' NPROC='' BATCHSIZE='' EPOCHS=''
DEVICE='' DATASET_OPT='' RESUME='' LR='' LR_BACKBONE='' WEIGHT_DECAY=''
CLIP_MAX_NORM='' COCO_PATH=''
MAX_TRAIN_STEPS='' MAX_SECONDS='' EVAL_IMAGES='' CHECKPOINT_EVERY=''
SEED=''
AUDIT_MODE=''

assign_value() {
    local flag=$1 val=$2
    case $flag in
        --output-dir|--output_dir)   OUTPUT_DIR=$val ;;
        --model_name)                MODEL_NAME=$val ;;
        --nproc_per_node)            NPROC=$val ;;
        --bs|--batch_size|--batchsize)
            if test -n "$BATCHSIZE" && test "$BATCHSIZE" != "$val"; then
                entry_fail "conflicting batch-size options: '$BATCHSIZE' vs '$val'"
            fi
            BATCHSIZE=$val ;;
        --epochs|--epoch)
            if test -n "$EPOCHS" && test "$EPOCHS" != "$val"; then
                entry_fail "conflicting epoch options: '$EPOCHS' vs '$val'"
            fi
            EPOCHS=$val ;;
        --device)                    DEVICE=$val ;;
        --dataset_path|--dataset-path|--model_root|--model-root)
            if test -n "$DATASET_OPT" && test "$DATASET_OPT" != "$val"; then
                entry_fail "conflicting --dataset_path/--model_root options: '$DATASET_OPT' vs '$val'"
            fi
            DATASET_OPT=$val ;;
        --resume)                    RESUME=$val ;;
        --seed)                      SEED=$val ;;
        --max-train-steps)           MAX_TRAIN_STEPS=$val ;;
        --max-seconds)               MAX_SECONDS=$val ;;
        --eval-images)               EVAL_IMAGES=$val ;;
        --checkpoint-every)          CHECKPOINT_EVERY=$val ;;
        --lr)                        LR=$val ;;
        --lr_backbone)               LR_BACKBONE=$val ;;
        --weight_decay)              WEIGHT_DECAY=$val ;;
        --clip_max_norm)             CLIP_MAX_NORM=$val ;;
        --coco_path|--coco-path)     COCO_PATH=$val ;;
        *)                           entry_fail "internal: unhandled value flag $flag" ;;
    esac
}

# --help is intercepted BEFORE _entry_env.sh is sourced, so bare-shell help
# never requires MODEL_ROOT, a vendor interpreter or any SDAA runtime.
for _pre in "$@"; do
    case $_pre in
        -h|--help|-help) usage; exit 0 ;;
    esac
done
unset _pre

while test "$#" -gt 0; do
    a=$1; shift
    case $a in
        --*=*)
            flag=${a%%=*}
            if value_flag "$flag"; then
                assign_value "$flag" "${a#*=}"
            elif test "$flag" = '--help'; then
                usage; exit 0
            else
                entry_fail "unknown option '${flag}' (the '=value' form is only valid for value flags)"
            fi ;;
        -h|--help|-help) usage; exit 0 ;;
        *)
            if value_flag "$a"; then
                test "$#" -gt 0 || entry_fail "option '$a' requires a value"
                assign_value "$a" "$1"; shift
            else
                case $a in
                    --eval)           EVAL=1 ;;
                    --audit-msda)     AUDIT_MODE=on ;;
                    --no-audit-msda)  AUDIT_MODE=off ;;
                    --)               : ;;
                    -*)               entry_fail "unknown option '$a'" ;;
                    *)                entry_fail "unexpected positional argument '$a'" ;;
                esac
            fi ;;
    esac
done
: "${EVAL:-}"

# --- supported-value validation ----------------------------------------------
test -z "$MODEL_NAME" || test "$MODEL_NAME" = DeformableDETR ||
    entry_fail "--model_name '$MODEL_NAME' is unsupported: this entry delivers DeformableDETR only"
case $NPROC in
    '') ;;
    1) ;;
    *) entry_fail "--nproc_per_node must be 1 (validated single-process single-card entry); got '$NPROC'" ;;
esac
case $DEVICE in
    ''|sdaa|sdaa:0) ;;
    *) entry_fail "--device must be 'sdaa' or 'sdaa:0' (single SDAA card; no CPU fallback); got '$DEVICE'" ;;
esac
if test -n "$BATCHSIZE"; then
    entry_uint "$BATCHSIZE" || entry_fail "--batchsize requires a positive integer, got '$BATCHSIZE'"
    test "$BATCHSIZE" -ge 1 || entry_fail "--batchsize must be >= 1, got '$BATCHSIZE'"
fi
if test -n "$EPOCHS"; then
    entry_uint "$EPOCHS" || entry_fail "--epoch requires a positive integer, got '$EPOCHS'"
    # The official R50 single-scale profile (including its 20-epoch schedule) is
    # part of the validated contract enforced by train_sdaa.py; reject anything
    # else here instead of failing late.  Bounded runs use --max-train-steps /
    # --max-seconds, which do not change the recipe.
    test "$EPOCHS" -eq 20 ||
        entry_fail "--epoch is fixed at 20 by the validated official R50 single-scale recipe (got '$EPOCHS'); use --max-train-steps/--max-seconds for bounded runs"
fi
test -z "$SEED" || entry_int "$SEED" ||
    entry_fail "--seed requires an integer, got '$SEED'"
for pair in "--max-train-steps|$MAX_TRAIN_STEPS" \
            "--eval-images|$EVAL_IMAGES" "--checkpoint-every|$CHECKPOINT_EVERY"; do
    name=${pair%%|*}; val=${pair#*|}
    test -z "$val" || entry_uint "$val" ||
        entry_fail "$name requires a nonnegative integer, got '$val'"
done
test -z "$MAX_SECONDS" || { entry_float "$MAX_SECONDS" && test "${MAX_SECONDS#-}" = "$MAX_SECONDS"; } ||
    entry_fail "--max-seconds requires a nonnegative number, got '$MAX_SECONDS'"
for pair in "--lr|$LR" "--lr_backbone|$LR_BACKBONE" "--weight_decay|$WEIGHT_DECAY"; do
    name=${pair%%|*}; val=${pair#*|}
    test -z "$val" || entry_float "$val" || entry_fail "$name requires a number, got '$val'"
done
test -z "$CLIP_MAX_NORM" || { entry_float "$CLIP_MAX_NORM" && test "${CLIP_MAX_NORM#-}" = "$CLIP_MAX_NORM"; } ||
    entry_fail "--clip_max_norm requires a nonnegative number, got '$CLIP_MAX_NORM'"

# --- environment defaults (fail-closed), BEFORE _entry_env.sh ----------------
if test -n "$DATASET_OPT"; then
    MODEL_ROOT=$DATASET_OPT
elif test -z "${MODEL_ROOT:-}"; then
    MODEL_ROOT=$DEFAULT_MODEL_ROOT
fi
if test ! -f "$MODEL_ROOT/$CKPT"; then
    if test -f "$MODEL_ROOT/data_ckpt/$CKPT"; then
        MODEL_ROOT=$MODEL_ROOT/data_ckpt
    else
        entry_fail "MODEL_ROOT '$MODEL_ROOT' contains neither '$CKPT' nor 'data_ckpt/$CKPT': point MODEL_ROOT (or --dataset_path/--model_root) at the provisioned read-only data_ckpt directory, e.g. $DEFAULT_MODEL_ROOT"
    fi
fi
test -d "$MODEL_ROOT/voc_coco" ||
    entry_fail "MODEL_ROOT '$MODEL_ROOT' must contain the read-only assets voc_coco/ and '$CKPT'; values given via --dataset_path/--model_root are accepted too, provided they resolve to the data_ckpt directory above"
export MODEL_ROOT
export PYTHON="${PYTHON:-$DEFAULT_PYTHON}"

# shellcheck source=_entry_env.sh
. "$SCRIPT_DIR/_entry_env.sh"
detr_setup_env "$PROJECT_DIR" "run_scripts/test.sh"
detr_stage_assets "$PROJECT_DIR"

cd "$SCRIPT_DIR"
PASS_ARGS=()
if test -n "$OUTPUT_DIR"; then      PASS_ARGS+=(--output-dir "$OUTPUT_DIR"); fi
if test -n "$BATCHSIZE"; then       PASS_ARGS+=(--batchsize "$BATCHSIZE"); fi
if test -n "$EPOCHS"; then          PASS_ARGS+=(--epoch "$EPOCHS"); fi
if test -n "$SEED"; then            PASS_ARGS+=(--seed "$SEED"); fi
if test -n "$DEVICE"; then          PASS_ARGS+=(--device "$DEVICE"); fi
if test -n "$RESUME"; then          PASS_ARGS+=(--resume "$RESUME"); fi
if test -n "$MAX_TRAIN_STEPS"; then PASS_ARGS+=(--max-train-steps "$MAX_TRAIN_STEPS"); fi
if test -n "$MAX_SECONDS"; then     PASS_ARGS+=(--max-seconds "$MAX_SECONDS"); fi
if test -n "$EVAL_IMAGES"; then     PASS_ARGS+=(--eval-images "$EVAL_IMAGES"); fi
if test -n "$CHECKPOINT_EVERY"; then PASS_ARGS+=(--checkpoint-every "$CHECKPOINT_EVERY"); fi
if test -n "${EVAL:-}"; then        PASS_ARGS+=(--eval); fi
case $AUDIT_MODE in
    on)  PASS_ARGS+=(--audit-msda) ;;
    off) PASS_ARGS+=(--no-audit-msda) ;;
esac
if test -n "$LR";             then PASS_ARGS+=(--lr "$LR"); fi
if test -n "$LR_BACKBONE";    then PASS_ARGS+=(--lr_backbone "$LR_BACKBONE"); fi
if test -n "$WEIGHT_DECAY";   then PASS_ARGS+=(--weight_decay "$WEIGHT_DECAY"); fi
if test -n "$CLIP_MAX_NORM";  then PASS_ARGS+=(--clip_max_norm "$CLIP_MAX_NORM"); fi
if test -n "$COCO_PATH";      then PASS_ARGS+=(--coco_path "$COCO_PATH"); fi
exec "$PYTHON" run_DeformableDETR.py "${PASS_ARGS[@]}"
