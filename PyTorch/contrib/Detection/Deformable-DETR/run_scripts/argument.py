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

"""ModelZoo arguments for the official VOC2007 single-scale training profile."""
import argparse


def parse_args():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--model_name', default='DeformableDETR', choices=['DeformableDETR'])
    parser.add_argument('--batchsize', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--resume', default='')
    parser.add_argument('--max-train-steps', type=int, default=0)
    parser.add_argument('--max-seconds', type=float, default=0)
    parser.add_argument('--eval-images', type=int, default=0)
    parser.add_argument('--checkpoint-every', type=int, default=1000)
    parser.add_argument('--eval', action='store_true')
    return parser.parse_args()
