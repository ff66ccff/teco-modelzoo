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

"""Run the validated native SDAA profile without invoking a shell."""
from pathlib import Path
import subprocess
import sys
from argument import parse_args

args = parse_args()
root = Path(__file__).resolve().parents[1]
python = Path('/home/py312/bin/python')
if Path(sys.executable).resolve() != python.resolve():
    raise RuntimeError('use /home/py312/bin/python')
command = [str(python), '-u', str(root / 'train_sdaa.py'), '--device', 'sdaa:0',
           '--batch_size', str(args.batchsize), '--epochs', str(args.epoch), '--seed', str(args.seed),
           '--output_dir', args.output_dir, '--max-train-steps', str(args.max_train_steps),
           '--max-seconds', str(args.max_seconds), '--eval-images', str(args.eval_images),
           '--checkpoint-every', str(args.checkpoint_every), '--audit-msda']
if args.resume:
    command.extend(['--resume', args.resume])
if args.eval:
    command.append('--eval')
raise SystemExit(subprocess.run(command, cwd=root, check=False).returncode)
