# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
# Copyright (c) 2026 Sichuan University contributors.

"""ModelZoo arguments for the official VOC2007 single-scale training profile.

run_scripts/test.sh normalizes the ModelZoo-style aliases (--bs/--batch_size ->
--batchsize, --epochs -> --epoch, --output_dir -> --output-dir, --device,
--model_name, --nproc_per_node, --dataset_path/--model_root), applies the
PYTHON/MODEL_ROOT defaults and defaults --output-dir to a fresh unique
directory BEFORE this parser runs; --output-dir is therefore no longer
mandatory.  The aliases are still declared here so a direct
``bash run_scripts/run_DeformableDETR.py ...`` call accepts them too.
"""
import argparse
import datetime
import os
import pathlib
import sys


def _default_output_dir():
    # New unique directory under <project>/outputs, so no flag is mandatory
    # and every invocation keeps the "fresh empty output dir" contract.
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = pathlib.Path(__file__).resolve().parents[1] / 'outputs' / f'run-{stamp}'
    if path.exists():
        path = path.with_name(f'{path.name}-{os.getpid()}')
    return str(path)


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_name', default='DeformableDETR', choices=['DeformableDETR'])
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', default=_default_output_dir())
    parser.add_argument('--resume', default='')
    parser.add_argument('--max-train-steps', type=int, default=0)
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('--eval-images', type=int, default=0)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--eval', action='store_true')
    # Native MSDA audit: enabled by default (run_DeformableDETR.py appends
    # --audit-msda unless --no-audit-msda was given).  --no-audit-msda maps
    # audit off; --audit-msda kept for explicitness (audit on, the default).
    parser.add_argument('--audit-msda', dest='audit_msda', action='store_true',
                        default=None)
    parser.add_argument('--no-audit-msda', dest='audit_msda', action='store_false',
                        default=None)

    # Normalization fallbacks for direct run_DeformableDETR.py invocations:
    # run_scripts/test.sh already validated them before reaching this parser
    # and does NOT forward --model_name/--nproc_per_node/--dataset_path.
    parser.add_argument('--bs', '--batch_size', dest='bs_alias', type=int, default=None)
    parser.add_argument('--epochs', dest='epochs_alias', type=int, default=None)
    parser.add_argument('--output_dir', dest='output_dir_alias', default=None)
    parser.add_argument('--device', dest='device_alias', default=None,
                        help="accepted values 'sdaa' or 'sdaa:0'; the runner is single-device")
    parser.add_argument('--nproc_per_node', dest='nproc_alias', type=int, default=None)
    parser.add_argument('--dataset_path', '--model_root', dest='dataset_alias', default=None,
                        help='informational here; test.sh resolves it and exports MODEL_ROOT')

    # Official-recipe-overriding passthrough supported by train_sdaa.py via
    # main.get_args_parser.  Defaults are None by design: the validated recipe
    # (lr=2e-5, lr_backbone=2e-6) must not be silently re-defaulted, so these
    # are forwarded to train_sdaa.py ONLY when the user passed them.
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--lr_backbone', type=float, default=None)
    parser.add_argument('--weight_decay', type=float, default=None)
    parser.add_argument('--clip_max_norm', type=float, default=None)
    parser.add_argument('--coco_path', default=None)

    args = parser.parse_args()
    if args.device_alias is not None and args.device_alias not in ('sdaa', 'sdaa:0'):
        sys.exit("ERROR: --device must be 'sdaa' or 'sdaa:0'; got '%s'" % args.device_alias)
    if args.nproc_alias is not None and args.nproc_alias != 1:
        sys.exit('ERROR: --nproc_per_node must be 1 (single-process single-card entry); '
                 'got %d' % args.nproc_alias)
    if args.bs_alias is not None:
        args.batchsize = args.bs_alias
    if args.epochs_alias is not None:
        args.epoch = args.epochs_alias
    if args.output_dir_alias is not None:
        args.output_dir = args.output_dir_alias
    # The official R50 single-scale profile (including its 20-epoch schedule) is
    # part of the validated contract enforced by train_sdaa.py; reject anything
    # else in the wrapper too, so a direct run_DeformableDETR.py invocation also
    # fails fast with the actionable hint instead of failing after model setup.
    if args.epoch != 20:
        sys.exit('ERROR: --epoch is fixed at 20 by the validated official R50 single-scale '
                 'recipe (got %d); use --max-train-steps/--max-seconds for bounded runs'
                 % args.epoch)
    _ensure_fresh_output_dir(args)
    return args


def _ensure_fresh_output_dir(args):
    out = pathlib.Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        sys.exit('ERROR: --output-dir %s already exists and is not empty; every invocation '
                 'must use a NEW empty output directory' % args.output_dir)
