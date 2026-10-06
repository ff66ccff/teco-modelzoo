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

# Copyright (c) 2026 Sichuan University contributors.
# Licensed under the Apache License, Version 2.0 (see vendor/LICENSE).
"""Single-device FP32 VOC training using the pinned official model and native MSDA."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch_sdaa  # registers the vendor device, without changing installed packages

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
from main import get_args_parser
from models import build_model
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate
import util.misc as utils
import tecoops
from tcap_dllogger import Logger, JSONStreamBackend, Verbosity

FORMAT = "scu-voc-sdaa-training-v1"
OFFICIAL_COMMIT = "1bda1f956a326d2601302413e4618c0fd109e799"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def configuration(args):
    # Only execution bounds, paths and device ordinal may change on resume.
    excluded = {"output_dir", "resume", "pretrained", "coco_path", "device", "eval",
                "max_train_steps", "max_seconds", "eval_images", "checkpoint_every",
                "audit_msda", "num_workers", "start_epoch"}
    return {k: v for k, v in vars(args).items() if k not in excluded}


def rng_state():
    numpy_state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "numpy": (numpy_state[0], torch.tensor(numpy_state[1].astype(np.int64)),
                      *numpy_state[2:]), "sdaa": torch.sdaa.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    algorithm, keys, position, has_gaussian, cached_gaussian = state["numpy"]
    np.random.set_state((algorithm, keys.numpy().astype(np.uint32), position,
                         has_gaussian, cached_gaussian))
    torch.sdaa.set_rng_state_all(state["sdaa"])


def assert_restored(original, restored):
    """Check the exact resume starting state before any new optimizer update."""
    if isinstance(original, torch.Tensor):
        if (original.shape != restored.shape or original.dtype != restored.dtype
                or not torch.equal(original.cpu(), restored.detach().cpu())):
            raise RuntimeError("training state changed while restoring")
        return 1
    if isinstance(original, dict):
        if original.keys() != restored.keys():
            raise RuntimeError("restored state keys differ")
        return sum(assert_restored(original[key], restored[key]) for key in original)
    if isinstance(original, (list, tuple)):
        if len(original) != len(restored):
            raise RuntimeError("restored state lengths differ")
        return sum(assert_restored(a, b) for a, b in zip(original, restored))
    if original != restored:
        raise RuntimeError("restored scalar state differs")
    return 0


def save_checkpoint(path, model, optimizer, scheduler, args, epoch, cursor, step, assets):
    temporary = path.with_name(path.name + ".tmp")
    torch.sdaa.synchronize()
    torch.save({"format": FORMAT, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch, "batch_cursor": cursor, "global_step": step,
                "rng": rng_state(), "configuration": configuration(args),
                "assets": assets, "official_commit": OFFICIAL_COMMIT}, temporary)
    temporary.replace(path)


def load_pretrained(path, model):
    # The supplied legacy checkpoint contains argparse.Namespace. No arbitrary
    # globals or URL loading is allowed. Its model weights are never written.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = {k: v for k, v in checkpoint["model"].items() if "class_embed" not in k}
    missing, unexpected = model.load_state_dict(state, strict=False)
    expected = {f"class_embed.{layer}.{part}" for layer in range(6)
                for part in ("weight", "bias")}
    if set(missing) != expected or unexpected:
        raise RuntimeError(f"pretrained contract mismatch: {missing=}, {unexpected=}")
    return {"missing_keys": missing, "unexpected_keys": unexpected,
            "classifier_reinitialized": True, "weights_only": True}


def make_optimizer(model, args):
    def matches(name, keywords):
        return any(keyword in name for keyword in keywords)
    groups = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad
                    and not matches(n, args.lr_backbone_names)
                    and not matches(n, args.lr_linear_proj_names)], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad
                    and matches(n, args.lr_backbone_names)], "lr": args.lr_backbone},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad
                    and matches(n, args.lr_linear_proj_names)],
         "lr": args.lr * args.lr_linear_proj_mult},
    ]
    parameters = [id(p) for group in groups for p in group["params"]]
    if len(parameters) != len(set(parameters)):
        raise RuntimeError("optimizer groups overlap")
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


def install_audit():
    # Optional instrumentation is selected once. The native model entry has no
    # backend selection or environment lookup on its forward path.
    counts = {"forward": 0, "backward": 0}
    forward, backward = tecoops.ms_deform_attn_forward, tecoops.ms_deform_attn_backward
    def audited_forward(*inputs):
        counts["forward"] += 1
        return forward(*inputs)
    def audited_backward(*inputs):
        counts["backward"] += 1
        return backward(*inputs)
    tecoops.ms_deform_attn_forward = audited_forward
    tecoops.ms_deform_attn_backward = audited_backward
    return counts


def batch_indices(length, batch_size, seed, epoch, cursor):
    generator = torch.Generator().manual_seed(seed + epoch)
    permutation = torch.randperm(length, generator=generator).tolist()
    usable = length // batch_size * batch_size
    return [permutation[index:index + batch_size]
            for index in range(cursor * batch_size, usable, batch_size)]


def main(args):
    if Path(sys.executable).resolve() != Path("/home/py312/bin/python").resolve():
        raise RuntimeError("use /home/py312/bin/python")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this validated entry supports one SDAA process only")
    if not args.device.startswith("sdaa") or not torch.sdaa.is_available():
        raise ValueError("native training requires an available SDAA device")
    if args.num_workers != 0:
        raise ValueError("exact augmentation RNG resume requires --num_workers 0")
    if (args.backbone, args.num_feature_levels, args.num_classes, args.epochs,
        args.enc_layers, args.dec_layers, args.hidden_dim, args.nheads,
        args.num_queries, args.enc_n_points, args.dec_n_points) != (
            "resnet50", 1, 21, 20, 6, 6, 256, 8, 300, 4, 4):
        raise ValueError("use the fixed official R50 single-scale 20-epoch model")
    if args.masks or args.two_stage or args.with_box_refine or args.sgd or not args.aux_loss:
        raise ValueError("this entry validates the official single-scale AdamW profile")
    if args.start_epoch or args.frozen_weights or args.cache_mode or args.lr_drop_epochs:
        raise ValueError("use --resume for training state; unsupported model variants are rejected")
    if not args.output_dir or args.batch_size < 1 or args.max_train_steps < 0 or args.max_seconds < 0:
        raise ValueError("explicit output directory and nonnegative execution bounds required")
    if args.eval and not args.resume:
        raise ValueError("--eval requires a trained --resume checkpoint")
    if args.dilation or args.dim_feedforward != 1024 or args.dropout != 0.1 or args.position_embedding != "sine":
        raise ValueError("the competition network and dropout configuration are fixed")
    args.distributed = False
    device = torch.device(args.device)
    torch.sdaa.set_device(device)
    torch.manual_seed(args.seed)
    torch.sdaa.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(4)
    output = Path(args.output_dir).resolve()
    root = Path(os.environ["MODEL_ROOT"]).resolve()
    pretrained = Path(args.pretrained).resolve()
    if not pretrained.is_relative_to(root) or not pretrained.is_file():
        raise ValueError("pretrained checkpoint must be a read-only MODEL_ROOT reference")
    if output == root or output.is_relative_to(root):
        raise ValueError("training output must be outside MODEL_ROOT")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh output directory for each invocation")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise OSError("at least 2 GiB free output space is required for atomic checkpoints")
    logger = Logger([JSONStreamBackend(Verbosity.VERBOSE, str(output / "sdaa.log"), append=False)])
    for name, unit, stage in (("train.loss", "", "TRAIN"), ("train.ips", "imgs/s", "TRAIN"),
                              ("val.loss", "", "VAL"), ("val.ips", "imgs/s", "VAL")):
        logger.metadata(name, {"unit": unit, "STAGE": stage})
    assets = {"pretrained_sha256": sha256(pretrained),
              "train_json_sha256": sha256(Path(args.coco_path) / "train/train.json"),
              "val_json_sha256": sha256(Path(args.coco_path) / "val/val.json")}
    expected_assets = {"pretrained_sha256": "d442fb2365d6e9640347b2b38686089d13cd55a5c3790d63e3602d079791fed1",
                       "train_json_sha256": "e511d3a0e39f45162887e4d9f0bed63bf8f01e4c421d965445185b72990f1ab5",
                       "val_json_sha256": "14edfb83a2525773c9e5b6e6d4bfa57a866de1cb3ca164c46afe0ac97629b561"}
    if assets != expected_assets:
        raise ValueError("official competition asset hashes do not match")
    counts = install_audit() if args.audit_msda else None
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    optimizer, scheduler = make_optimizer(model, args)
    dataset_train = build_dataset("train", args)
    dataset_val = build_dataset("val", args)
    if len(dataset_train) != 5011 or len(dataset_val) != 4952:
        raise ValueError("use the complete official VOC2007 datasets")
    if args.batch_size > len(dataset_train):
        raise ValueError("batch size exceeds the training set")
    epoch = cursor = global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if (checkpoint.get("format") != FORMAT or checkpoint["configuration"] != configuration(args)
                or checkpoint["assets"] != assets or checkpoint["official_commit"] != OFFICIAL_COMMIT):
            raise ValueError("resume checkpoint/configuration mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        epoch, cursor, global_step = (checkpoint[k] for k in ("epoch", "batch_cursor", "global_step"))
        restore_rng(checkpoint["rng"])
        verified = {"model": assert_restored(checkpoint["model"], model.state_dict()),
                    "optimizer": assert_restored(checkpoint["optimizer"], optimizer.state_dict()),
                    "scheduler": assert_restored(checkpoint["lr_scheduler"], scheduler.state_dict()),
                    "rng": assert_restored(checkpoint["rng"], rng_state())}
        load_report = {"classifier_reinitialized": False, "optimizer_restored": True,
                       "scheduler_restored": True, "rng_restored": True,
                       "epoch": epoch, "batch_cursor": cursor, "global_step": global_step,
                       "exact_starting_state_tensor_counts": verified}
    else:
        load_report = load_pretrained(pretrained, model)
    if not 0 <= epoch <= 20 or not 0 <= cursor <= len(dataset_train) // args.batch_size:
        raise ValueError("invalid resume epoch or minibatch cursor")
    extension = next(Path(tecoops.__file__).parent.glob("_torch_ext*.so"))
    cores = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if "libteco_ops.so" in line})
    receipt = {"official_commit": OFFICIAL_COMMIT, "assets": assets,
               "entry_sha256": sha256(Path(__file__)),
               "vendor_manifest_sha256": sha256(HERE / "vendor/SOURCE.json"),
               "configuration": configuration(args), "load": load_report,
               "native_extension": str(extension), "native_extension_sha256": sha256(extension),
               "mapped_cores": {path: sha256(path) for path in cores},
               "tecoops_file": tecoops.__file__, "dataset_counts": [5011, 4952],
               "python": str(Path(sys.executable).resolve()), "torch": torch.__version__}
    atomic_json(output / "startup.json", receipt)
    print(json.dumps({"startup": receipt}), flush=True)
    torch.sdaa.synchronize()
    torch.sdaa.reset_peak_memory_stats()
    started = time.monotonic()
    invocation_steps = 0
    completed_epochs = []
    last_loss = None
    last_evaluation = None
    stop = args.eval
    with (output / "steps.jsonl").open("x", buffering=1) as log:
        while epoch < args.epochs and not stop:
            batches = batch_indices(len(dataset_train), args.batch_size, args.seed, epoch, cursor)
            loader = torch.utils.data.DataLoader(dataset_train, batch_sampler=batches,
                collate_fn=utils.collate_fn, num_workers=0, pin_memory=False,
                generator=torch.Generator().manual_seed(args.seed + epoch))
            model.train()
            criterion.train()
            step_started = time.monotonic()
            for samples, targets in loader:
                before = dict(counts) if counts is not None else None
                ids = [int(target["image_id"].item()) for target in targets]
                input_shape = list(samples.tensors.shape)
                samples = samples.to(device)
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
                outputs = model(samples)
                if outputs["pred_logits"].shape != (args.batch_size, 300, 21):
                    raise RuntimeError("unexpected classifier output")
                loss_dict = criterion(outputs, targets)
                loss = sum(loss_dict[k] * criterion.weight_dict[k]
                           for k in loss_dict if k in criterion.weight_dict)
                last_loss = float(loss.detach().cpu())
                if not math.isfinite(last_loss):
                    raise FloatingPointError(f"non-finite loss: {last_loss}")
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                grad_norm = float(norm.detach().cpu())
                if not math.isfinite(grad_norm):
                    raise FloatingPointError("non-finite gradient norm")
                optimizer.step()
                torch.sdaa.synchronize()
                cursor += 1
                global_step += 1
                invocation_steps += 1
                row = {"epoch": epoch, "batch_cursor": cursor, "global_step": global_step,
                       "image_ids": ids, "input_shape": input_shape, "loss": last_loss,
                       "grad_norm": grad_norm, "lr": [g["lr"] for g in optimizer.param_groups],
                       "elapsed_seconds": time.monotonic() - started}
                logger.log(step=(epoch, global_step), data={"rank": 0, "bs": args.batch_size,
                    "train.loss": last_loss, "train.ips": args.batch_size / (time.monotonic() - step_started)})
                if counts is not None:
                    row["native_calls"] = {k: counts[k] - before[k] for k in counts}
                    if row["native_calls"] != {"forward": 12, "backward": 12}:
                        raise RuntimeError(f"unpaired native model path: {row['native_calls']}")
                log.write(json.dumps(row, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
                if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                    save_checkpoint(output / "checkpoint.pth", model, optimizer, scheduler,
                                    args, epoch, cursor, global_step, assets)
                stop = ((args.max_train_steps and invocation_steps >= args.max_train_steps)
                        or (args.max_seconds and time.monotonic() - started >= args.max_seconds))
                if stop:
                    break
                step_started = time.monotonic()
            if cursor == len(dataset_train) // args.batch_size:
                scheduler.step()
                completed_epochs.append(epoch)
                epoch += 1
                cursor = 0
                save_checkpoint(output / "checkpoint.pth", model, optimizer, scheduler,
                                args, epoch, cursor, global_step, assets)
                if not stop:
                    last_evaluation = evaluate_model(model, criterion, postprocessors, dataset_val, args, device, output, logger,
                                   f"evaluation-epoch-{epoch:02d}.json")
    training_seconds = time.monotonic() - started
    if not args.eval:
        save_checkpoint(output / "checkpoint.pth", model, optimizer, scheduler,
                        args, epoch, cursor, global_step, assets)
    # A bounded run evaluates the requested subset explicitly. Only the default
    # full evaluator may produce a competition AP result.
    if epoch == 20 and last_evaluation is not None:
        evaluation = last_evaluation
        atomic_json(output / "evaluation.json", evaluation)
    else:
        evaluation = evaluate_model(model, criterion, postprocessors, dataset_val, args, device,
                                    output, logger, "evaluation.json")
    result = {"status": "evaluation_complete" if args.eval else
              "full_20_epochs_complete" if epoch == 20 else "bounded_training_checkpoint",
              "epoch": epoch, "batch_cursor": cursor, "global_step": global_step,
              "invocation_steps": invocation_steps, "completed_epochs": completed_epochs,
              "training_seconds": training_seconds, "last_loss": last_loss,
              "evaluation": evaluation, "native_calls": counts,
              "peak_allocated_mib": torch.sdaa.max_memory_allocated() / (1024 * 1024),
              "peak_reserved_mib": torch.sdaa.max_memory_reserved() / (1024 * 1024),
              "checkpoint": None if args.eval else str(output / "checkpoint.pth"),
              "limits": ["single-device FP32 eager first-order MSDA; no AMP/DDP claim",
                         "official CPU Hungarian matching is retained",
                         "subset AP is diagnostic and does not meet the full test gate"]}
    atomic_json(output / "result.json", result)
    logger.flush()
    print(json.dumps({"result": result}, allow_nan=False), flush=True)


def evaluate_model(model, criterion, postprocessors, dataset, args, device, output, logger, name):
    count = args.eval_images or len(dataset)
    if not 0 < count <= len(dataset):
        raise ValueError("invalid --eval-images")
    indices = range(count)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices),
        batch_size=args.batch_size, shuffle=False, collate_fn=utils.collate_fn,
        num_workers=0, pin_memory=False,
        generator=torch.Generator().manual_seed(args.seed))
    saved_rng = rng_state()
    started = time.monotonic()
    try:
        stats, evaluator = evaluate(model, criterion, postprocessors, loader,
                                    get_coco_api_from_dataset(dataset), device, str(output))
        torch.sdaa.synchronize()
    finally:
        # Validation's fixed resize still consumes Python RNG. Keep evaluation
        # from changing the subsequent training augmentation sequence.
        restore_rng(saved_rng)
    logger.log(step="evaluation", data={"rank": 0, "val.loss": stats["loss"],
                                       "val.ips": count / (time.monotonic() - started)})
    result = {"images": count, "full_test": count == 4952, "stats": stats}
    atomic_json(output / name, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__, parents=[get_args_parser()])
    parser.set_defaults(device="sdaa:0", epochs=20, num_feature_levels=1, num_classes=21,
                        lr=2e-5, lr_backbone=2e-6, lr_drop=14, num_workers=0,
                        coco_path="./data/voc_coco")
    parser.add_argument("--pretrained", default="./r50_deformable_detr_single_scale-checkpoint.pth")
    parser.add_argument("--max-train-steps", type=int, default=0,
                        help="bound this invocation; zero runs to the total 20-epoch target")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="stop after a completed minibatch; zero has no time bound")
    parser.add_argument("--eval-images", type=int, default=0,
                        help="diagnostic prefix; zero evaluates all 4952 test images")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--audit-msda", action="store_true")
    main(parser.parse_args())
