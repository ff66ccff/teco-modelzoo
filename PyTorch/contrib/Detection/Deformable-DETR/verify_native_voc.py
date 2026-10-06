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

"""Focused native forward/gradient oracle at actual official VOC model boundaries."""
import argparse
import json
from pathlib import Path
import random
import sys
import numpy as np
import torch
import torch_sdaa

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_sdaa as training
from models.ops.modules.ms_deform_attn import MSDeformAttnFunction
from models.ops.functions.ms_deform_attn_func import ms_deform_attn_core_pytorch
import tecoops

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
profile = training.get_args_parser().parse_args([])
profile.device, profile.num_classes, profile.num_feature_levels = 'sdaa:0', 21, 1
profile.lr, profile.lr_backbone = 2e-5, 2e-6
profile.coco_path = str(HERE / 'data/voc_coco')
torch.sdaa.set_device(0)
torch.set_num_threads(4)
torch.manual_seed(42)
torch.sdaa.manual_seed_all(42)
random.seed(42)
np.random.seed(42)
model, criterion, unused = training.build_model(profile)
training.load_pretrained(HERE / 'r50_deformable_detr_single_scale-checkpoint.pth', model)
model.to('sdaa')
model.train()
dataset = training.build_dataset('train', profile)
samples, targets = training.utils.collate_fn([dataset[0], dataset[1]])
image_ids = [int(t['image_id']) for t in targets]
samples = samples.to('sdaa')
targets = [{k:v.to('sdaa') for k,v in t.items()} for t in targets]
original = MSDeformAttnFunction.apply
captures = {}
calls = 0
def capture(value, shapes, starts, locations, weights, step):
    global calls
    calls += 1
    output = original(value, shapes, starts, locations, weights, step)
    role = 'encoder' if locations.shape[1] == value.shape[1] else 'decoder'
    if role not in captures:
        record = {'inputs':[x.detach().cpu().clone() for x in (value, shapes, locations, weights)],
                  'output':output.detach().cpu().clone()}
        captures[role] = record
        def gradient_hook(gradient):
            record['grad_output'] = gradient.detach().cpu().clone()
        output.register_hook(gradient_hook)
    return output
MSDeformAttnFunction.apply = staticmethod(capture)
outputs = model(samples)
losses = criterion(outputs, targets)
loss = sum(value * criterion.weight_dict[name] for name,value in losses.items()
           if name in criterion.weight_dict)
loss.backward()
assert calls == 12 and set(captures) == {'encoder','decoder'}
gradient_count = 0
for parameter in model.parameters():
    if parameter.grad is not None:
        assert torch.isfinite(parameter.grad.detach().cpu()).all()
        gradient_count += 1
checks = {}
for role, capture in captures.items():
    value, shapes, locations, weights = capture['inputs']
    reference_inputs = [value.clone().requires_grad_(), shapes,
                        locations.clone().requires_grad_(), weights.clone().requires_grad_()]
    reference = ms_deform_attn_core_pytorch(*reference_inputs)
    torch.testing.assert_close(capture['output'], reference, atol=2e-4, rtol=2e-4)
    reference.backward(capture['grad_output'])
    native_inputs = [value.to('sdaa').requires_grad_(), shapes.to('sdaa'),
                     locations.to('sdaa').requires_grad_(), weights.to('sdaa').requires_grad_()]
    native = tecoops.ms_deform_attn(*native_inputs)
    native.backward(capture['grad_output'].to('sdaa'))
    comparison = {'forward':float((capture['output']-reference.detach()).abs().max())}
    for name, index in [('value',0),('locations',2),('weights',3)]:
        result = native_inputs[index].grad.detach().cpu()
        oracle = reference_inputs[index].grad
        torch.testing.assert_close(result, oracle, atol=2e-5, rtol=1e-3)
        comparison['grad_'+name] = float((result-oracle).abs().max())
    checks[role] = {'shapes':[list(x.shape) for x in capture['inputs']], 'max_abs':comparison}
report = {'status':'pass', 'official_commit':training.OFFICIAL_COMMIT,
          'image_ids':image_ids, 'model_native_forward_calls':calls,
          'finite_parameter_gradient_count':gradient_count, 'weighted_loss':float(loss.detach().cpu()),
          'checks':checks, 'forward_tolerance':{'atol':2e-4,'rtol':2e-4},
          'gradient_tolerance':{'atol':2e-5,'rtol':1e-3},
          'scope':'actual model boundaries; CPU oracle is explicit and never a model fallback'}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
