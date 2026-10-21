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
"""Distributed FP32 VOC training (real DDP over the card's four core groups).

太初卡是"单卡四核组"结构：一张卡由 4 个核心组（逻辑设备 sdaa:0..3）组成，
官方 r50 参考入口用的就是 "--nproc_per_node 4"。本入口用真实的 DDP=4：
4 个进程各占一个核心组、各自跑一份完整模型副本，梯度由 TCCL all-reduce 同步，
因此整卡吞吐是真的四路并行，而不是把单进程吞吐人工除以 4。TCAP 日志里
"单核心组"视图 = 本进程真实测到的吞吐，"aggregate" 视图 = 4 个进程合计
（= 单核组 x 4，因为四路真的同时在跑）。
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendor"))

# Capability contract, evaluated once at import time so every path into native
# training (train_sdaa.py, run_DeformableDETR.py, verify_native_voc.py) is gated
# on real capabilities: Python 3.11/3.12, torch/torchvision, a usable SDAA device
# and the paired tecoops package resolved under TECOOPS_API_ROOT (an older
# site-packages wheel can no longer be used silently).  It runs before the heavy
# stack imports so a wrong interpreter fails with an actionable message.
import runtime_contract

RUNTIME_RECEIPT = runtime_contract.require_capabilities(
    role="train_sdaa.py", require_tecoops_root=True)

import numpy as np
import torch
import torch_sdaa  # registers the vendor device, without changing installed packages
import torch.distributed as dist
from main import get_args_parser
from models import build_model
from datasets import build_dataset, get_coco_api_from_dataset
from datasets.coco_eval import CocoEvaluator
from engine import evaluate
import util.misc as utils
import tecoops
from tcap_dllogger import Logger, StdOutBackend, JSONStreamBackend, Verbosity

FORMAT = "scu-voc-sdaa-training-v1"
OFFICIAL_COMMIT = "1bda1f956a326d2601302413e4618c0fd109e799"

# 太初卡 = 单卡四核组（torch.sdaa.device_count() == 4）。DDP world size 必须是
# 1（单核组）或 4（整卡四核组）。测试机每个核心组有独立的 ~14.9 GiB 显存，
# 4 个副本不会互相挤占。
SDAA_CORE_GROUPS = 4


# ---------------------------------------------------------------------------
# Organizer TCAP stdout logging.
#
# 本入口跑真实 DDP：每个 rank 占一个核心组，并发跑自己的 minibatch，梯度经
# TCCL all-reduce 同步。因此日志里的"单核心组"视图就是该核心组**实测**的吞吐
# （bs = 每进程 batch，ips = 该进程每秒处理图像数），aggregate 视图是 4 路合计。
# 旧版本"单进程跑 + ips/4、bs/4"的换算被删除：那在四核组卡上把真实吞吐人为
# 缩小 4 倍，既不合理也不可比。每步输出两行（同一 step）：
#   TCAPDLL ... rank : 0  bs : 2  train.loss : <global mean>  train.ips : <per-cg> imgs/s
#   TCAPDLL ... rank : 0  scope : aggregate  bs : 8  train.loss : ...  train.ips : <4-way> imgs/s
# train.loss 是全局逐图均值（跨 rank all-reduce），两种视图同值。
# ---------------------------------------------------------------------------

def tcap_stdout_prefix(rank, timestamp):
    """Organizer front-end marker: ``TCAPDLL <timestamp> - `` (task-15 sample format).

    The vendored tcap_dllogger default prefix is ``TCAPPDLL`` (logger.py
    ``default_prefix_format``), while the organizer's expected front-end line is
    ``TCAPDLL ...``.  stdout must match the organizer's example, so an explicit
    prefix_format is passed to StdOutBackend; the JSON stream keeps the tool's own
    marker and the vendored ``tcap_dllogger/`` package is not modified.
    """
    return "TCAPDLL {} - ".format(timestamp)

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


def run_configuration(args, world):
    """configuration() + 并行形态：world size 与全局 batch 必须与 checkpoint 一致."""
    config = configuration(args)
    config["world_size"] = world
    config["global_batch_size"] = args.batch_size * world
    return config


# --eval 只做前向评估、不更新训练状态，因此允许用单进程评估 DDP 训练出的
# checkpoint：并行形态 / 优化器超参 / 批次规模都不参与一致性比较，只比较
# 网络与损失配方（这些变了，权重就没有可比性）。
EVAL_PARALLEL_KEYS = {"world_size", "global_batch_size", "batch_size", "nproc_per_node",
                      "distributed", "lr", "lr_backbone", "weight_decay", "lr_drop",
                      "lr_drop_epochs", "clip_max_norm", "seed"}


def comparable_config(config, for_eval):
    if not for_eval:
        return config
    return {key: value for key, value in config.items() if key not in EVAL_PARALLEL_KEYS}


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


def save_checkpoint(path, raw_model, optimizer, scheduler, args, epoch, cursor, step,
                    assets, rank, world, tag):
    """rank0 落盘；RNG 状态按 rank 收集，保证 DDP 断点续训可精确回到同一状态."""
    torch.sdaa.synchronize()
    rng = rng_state()
    if world > 1:
        rng = gather_payloads(rng, path.parent, "rng-" + tag, rank, world)
        if rank != 0:
            return
    temporary = path.with_name(path.name + ".tmp")
    torch.save({"format": FORMAT, "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(), "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch, "batch_cursor": cursor, "global_step": step,
                "rng": rng, "world_size": world,
                "configuration": run_configuration(args, world),
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


class NullLogger:
    """非 rank0 进程的日志占位：stdout/JSON 只由 rank0 写，集合通信仍由所有 rank 参与."""

    def metadata(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass

    def flush(self):
        pass


def distributed_env():
    """torchrun 注入的 (rank, world_size, local_rank)；单进程直接运行时为 (0, 1, 0)."""
    return (int(os.environ.get("RANK", "0")),
            int(os.environ.get("WORLD_SIZE", "1")),
            int(os.environ.get("LOCAL_RANK", "0")))


def init_distributed(world):
    """太初卡内四个核心组之间用 TCCL 做梯度 all-reduce（上游 r50 参考同一后端）."""
    if world < 2:
        return
    dist.init_process_group(backend="tccl", init_method="env://")
    # util.misc.get_local_size()/get_local_rank() 由上游 DETR 的
    # init_distributed_mode() 从这些环境变量取值；torchrun 只提供 LOCAL_RANK，
    # 不提供 LOCAL_SIZE，这里补齐后再让数据集构建读到正确的 shard 视图。
    # 注意：不要在这里放 dist.barrier() —— 本机 TCCL 的 barrier 不是可靠的
    # 全局屏障（见 gather_payloads 的说明），init_process_group 自带 rendezvous。
    os.environ.setdefault("LOCAL_SIZE", str(world))


def broadcast_decision(value, device, world):
    """把 rank 本地的分支判断变成所有 rank 一致的判断，避免集合通信步调不一致."""
    if world < 2:
        return bool(value)
    flag = torch.tensor([1.0 if value else 0.0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item() > 0.0)


def gather_payloads(payload, directory, tag, rank, world, timeout=1800.0):
    """用 output 目录 + 文件轮询做任意对象的 rank 间交换（替代 all_gather_object）。

    两个已知问题让这里不能走集合通信：
    1) torch.distributed.all_gather_object 在本机 SDAA 后端上按错误长度分配
       （实测 resize_ 请求 3.7e12 GiB 后 OOM）；
    2) 本机 TCCL 的 dist.barrier() 实测不是可靠的全局屏障 —— 空 shard 的 rank
       会在其它 rank 尚未到达时就"完成"（见 trace-ddp4 实录：rank2/3 先过，
       rank0/1 永久挂住），所以不能拿它做跨进程次序。
    改为：每个 rank 原子落盘（tmp + rename），rank0 轮询等待全部文件出现后读回。
    """
    path = directory / ("gather-%s-rank%d.pt" % (tag, rank))
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    if rank != 0:
        return None
    paths = [directory / ("gather-%s-rank%d.pt" % (tag, other)) for other in range(world)]
    deadline = time.monotonic() + timeout
    while True:
        missing = [item.name for item in paths if not item.is_file()]
        if not missing:
            break
        if time.monotonic() > deadline:
            raise TimeoutError("gather %s: missing payloads %s" % (tag, missing))
        time.sleep(0.05)
    return [torch.load(item, map_location="cpu", weights_only=False) for item in paths]


def mean_over_ranks(value, device, world):
    """各 rank 局部逐图均值再平均 = 全局逐图均值（每个 rank 的 batch 等大）."""
    if world < 2:
        return float(value)
    packed = torch.tensor([float(value), 1.0], device=device)
    dist.all_reduce(packed)
    return float((packed[0] / world).item())


def epoch_global_batches(length, batch_size, world, seed, epoch):
    """每个 epoch 的确定性全局 batch 切分（同一置换与 world size 无关）."""
    generator = torch.Generator().manual_seed(seed + epoch)
    permutation = torch.randperm(length, generator=generator).tolist()
    global_batch = batch_size * world
    usable = length // global_batch * global_batch
    return [permutation[start:start + global_batch]
            for start in range(0, usable, global_batch)]


def rank_batches(global_batches, batch_size, rank, cursor):
    """本 rank 在每个全局 batch 里拿到自己的等大 shard（互不重叠、覆盖全集）."""
    return [batch[rank * batch_size:(rank + 1) * batch_size]
            for batch in global_batches[cursor:]]


def main(args):
    # Interpreter capabilities are enforced by the import-time capability
    # contract above; the SDAA device itself is re-checked here.
    rank, world, local_rank = distributed_env()
    if args.nproc_per_node not in (1, SDAA_CORE_GROUPS):
        raise ValueError("--nproc-per-node must be 1 (one core group) or %d (the four core "
                         "groups of the card), got %d" % (SDAA_CORE_GROUPS, args.nproc_per_node))
    if world != args.nproc_per_node:
        raise ValueError("torchrun WORLD_SIZE=%d does not match --nproc-per-node=%d; launch "
                         "DDP through run_scripts/test.sh" % (world, args.nproc_per_node))
    if not str(args.device).startswith("sdaa") or not torch.sdaa.is_available():
        raise ValueError("native training requires an available SDAA device")
    if world > 1 and torch.sdaa.device_count() < world:
        raise ValueError("DDP world_size %d needs %d core groups, the runtime reports %d"
                         % (world, world, torch.sdaa.device_count()))
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
    args.distributed = world > 1
    if world > 1:
        # 每个 rank 绑定自己的核心组；--device 只表达"这张卡"，不表达序号。
        device = torch.device("sdaa", local_rank)
    else:
        device = torch.device(args.device)
        if device.type == "sdaa" and device.index is None:
            device = torch.device("sdaa", 0)
    torch.sdaa.set_device(device)
    init_distributed(world)
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
    # rank0 独占输出目录的创建与新鲜度检查：DDP 下其它 rank 若也检查，会和 rank0
    # 刚写下的 startup.json 竞争并误报"目录非空"。裁决用广播同步给所有 rank。
    decision = [None]
    if rank == 0:
        if output.exists() and any(output.iterdir()):
            decision = ["FileExistsError: use a fresh output directory for each invocation"]
        else:
            output.mkdir(parents=True, exist_ok=True)
            if shutil.disk_usage(output).free < 2 * 1024**3:
                decision = ["OSError: at least 2 GiB free output space is required for "
                            "atomic checkpoints"]
    if world > 1:
        dist.broadcast_object_list(decision, src=0)
    if decision[0]:
        raise RuntimeError(decision[0])
    # StdOutBackend MUST come first: the acceptance reads the front-end TCAP lines
    # from "bash run_scripts/test.sh" stdout; the JSON stream is only a by-product.
    # Only rank0 owns the logger: 4 processes must not write the same stdout/JSON file.
    if rank == 0:
        logger = Logger([StdOutBackend(Verbosity.DEFAULT, prefix_format=tcap_stdout_prefix),
                         JSONStreamBackend(Verbosity.VERBOSE, str(output / "sdaa.log"), append=False)])
        # fail-closed: never continue without a verifiable stdout backend, otherwise the
        # front-end log is silently missing again.
        if not any(isinstance(b, StdOutBackend) for b in logger.backends):
            raise RuntimeError("TCAP fail-closed: StdOutBackend is not in the logger backend list")
        for name, unit, stage, goal in (("train.loss", "", "TRAIN", "MINIMIZE"),
                                        ("train.ips", "imgs/s", "TRAIN", "MAXIMIZE"),
                                        ("bs", "", "TRAIN", "MAXIMIZE"),
                                        ("val.loss", "", "VAL", "MINIMIZE"),
                                        ("val.ips", "imgs/s", "VAL", "MAXIMIZE")):
            logger.metadata(name, {"unit": unit, "STAGE": stage, "GOAL": goal})
    else:
        logger = NullLogger()
    assets = {"pretrained_sha256": sha256(pretrained),
              "train_json_sha256": sha256(Path(args.coco_path) / "train/train.json"),
              "val_json_sha256": sha256(Path(args.coco_path) / "val/val.json")}
    expected_assets = {"pretrained_sha256": "d442fb2365d6e9640347b2b38686089d13cd55a5c3790d63e3602d079791fed1",
                       "train_json_sha256": "e511d3a0e39f45162887e4d9f0bed63bf8f01e4c421d965445185b72990f1ab5",
                       "val_json_sha256": "14edfb83a2525773c9e5b6e6d4bfa57a866de1cb3ca164c46afe0ac97629b561"}
    if assets != expected_assets:
        raise ValueError("official competition asset hashes do not match")
    # 官方单核组配方 lr=2e-5 / lr_backbone=2e-6；DDP 的全局 batch 随 world 线性
    # 放大，因此默认 lr 也按 linear scaling rule 缩放；world=1 时与旧版完全一致。
    # 显式传入 --lr / --lr_backbone 时不缩放。
    if args.lr is None:
        args.lr = 2e-5 * world
        args.lr_backbone = 2e-6 * world
    elif args.lr_backbone is None:
        args.lr_backbone = args.lr * 0.1
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
    # 模型初始权重在上面的同 seed 下各 rank 完全一致（DDP 不做参数广播）；此后把
    # 数据增强 / dropout 的 RNG 按 rank 分流，避免 4 个副本产生同样的随机流。
    if world > 1:
        rank_seed = args.seed + 1000 * (rank + 1)
        torch.manual_seed(rank_seed)
        torch.sdaa.manual_seed_all(rank_seed)
        np.random.seed(rank_seed)
        random.seed(rank_seed)
    raw_model = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            raw_model, device_ids=[local_rank], output_device=local_rank,
            broadcast_buffers=False, find_unused_parameters=False)
    epoch = cursor = global_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        expected = run_configuration(args, world)
        stored = dict(checkpoint.get("configuration") or {})
        if args.eval:
            expected = comparable_config(expected, True)
            stored = comparable_config(stored, True)
        if (checkpoint.get("format") != FORMAT or stored != expected
                or checkpoint["assets"] != assets or checkpoint["official_commit"] != OFFICIAL_COMMIT):
            raise ValueError("resume checkpoint/configuration mismatch: world_size, batch_size and "
                             "the official recipe must match the checkpoint; single-process "
                             "checkpoints carry no world size and are rejected for DDP resume")
        raw_model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        epoch, cursor, global_step = (checkpoint[k] for k in ("epoch", "batch_cursor", "global_step"))
        rng_payload = checkpoint["rng"]
        if args.eval:
            my_rng = None
        elif isinstance(rng_payload, list):
            if len(rng_payload) != world:
                raise ValueError("checkpoint stores %d RNG states but world_size is %d"
                                 % (len(rng_payload), world))
            my_rng = rng_payload[rank]
        else:
            if world > 1:
                raise ValueError("a single-process checkpoint cannot be resumed with world_size>1")
            my_rng = rng_payload
        if my_rng is not None:
            restore_rng(my_rng)
        verified = {"model": assert_restored(checkpoint["model"], raw_model.state_dict()),
                    "optimizer": assert_restored(checkpoint["optimizer"], optimizer.state_dict()),
                    "scheduler": assert_restored(checkpoint["lr_scheduler"], scheduler.state_dict())}
        if my_rng is not None:
            verified["rng"] = assert_restored(my_rng, rng_state())
        load_report = {"classifier_reinitialized": False, "optimizer_restored": True,
                       "scheduler_restored": True, "rng_restored": my_rng is not None,
                       "epoch": epoch, "batch_cursor": cursor, "global_step": global_step,
                       "exact_starting_state_tensor_counts": verified}
    else:
        load_report = load_pretrained(pretrained, raw_model)
    epoch_length = len(dataset_train) // (args.batch_size * world)
    if not 0 <= epoch <= 20 or not 0 <= cursor <= epoch_length:
        raise ValueError("invalid resume epoch or batch cursor")
    extension = next(Path(tecoops.__file__).parent.glob("_torch_ext*.so"))
    cores = sorted({line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
                    if "libteco_ops.so" in line})
    receipt = {"official_commit": OFFICIAL_COMMIT, "assets": assets,
               "entry_sha256": sha256(Path(__file__)),
               "vendor_manifest_sha256": sha256(HERE / "SOURCE.json"),
               "configuration": run_configuration(args, world), "load": load_report,
               "native_extension": str(extension), "native_extension_sha256": sha256(extension),
               "mapped_cores": {path: sha256(path) for path in cores},
               "tecoops_file": tecoops.__file__, "dataset_counts": [5011, 4952],
               "rank": rank, "world_size": world, "local_rank": local_rank,
               "device": str(device),
               "python": str(Path(sys.executable).resolve()), "torch": torch.__version__}
    if rank == 0:
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
    log_context = ((output / "steps.jsonl").open("x", buffering=1) if rank == 0
                   else contextlib.nullcontext())
    with log_context as log:
        while epoch < args.epochs and not stop:
            global_batches = epoch_global_batches(len(dataset_train), args.batch_size, world,
                                                 args.seed, epoch)
            loader = torch.utils.data.DataLoader(dataset_train,
                batch_sampler=rank_batches(global_batches, args.batch_size, rank, cursor),
                collate_fn=utils.collate_fn, num_workers=0, pin_memory=False,
                generator=torch.Generator().manual_seed(args.seed + epoch))
            model.train()
            criterion.train()
            for samples, targets in loader:
                before = dict(counts) if counts is not None else None
                iter_started = time.monotonic()
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
                local_loss = float(loss.detach().cpu())
                if not math.isfinite(local_loss):
                    raise FloatingPointError("non-finite local loss: %r" % local_loss)
                optimizer.zero_grad()
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                grad_norm = float(norm.detach().cpu())
                if not math.isfinite(grad_norm):
                    raise FloatingPointError("non-finite gradient norm")
                optimizer.step()
                torch.sdaa.synchronize()
                # 全局逐图 loss 与整卡步时间：所有 rank 在同一个 all-reduce 里对齐。
                last_loss = mean_over_ranks(local_loss, device, world)
                if not math.isfinite(last_loss):
                    raise FloatingPointError("non-finite global loss: %r" % last_loss)
                cursor += 1
                global_step += 1
                invocation_steps += 1
                iter_seconds = max(time.monotonic() - iter_started, 1e-9)
                native_calls = None
                if counts is not None:
                    native_calls = {k: counts[k] - before[k] for k in counts}
                    if native_calls != {"forward": 12, "backward": 12}:
                        raise RuntimeError("unpaired native model path: %s" % native_calls)
                if rank == 0:
                    row = {"epoch": epoch, "batch_cursor": cursor, "global_step": global_step,
                           "rank": 0, "world_size": world, "image_ids": ids,
                           "input_shape": input_shape, "loss": last_loss,
                           "loss_local": local_loss, "grad_norm": grad_norm,
                           "lr": [g["lr"] for g in optimizer.param_groups],
                           "step_seconds": iter_seconds,
                           "elapsed_seconds": time.monotonic() - started}
                    if native_calls is not None:
                        row["native_calls"] = native_calls
                    # TCAP: 单核心组视图 = 本进程实测（每进程 batch / 整卡步时间）；
                    # 聚合视图 = 4 个进程合计。这里没有人工除法：4 路是真在同时跑。
                    logger.log(step=(epoch, global_step),
                               data={"rank": 0, "bs": args.batch_size,
                                     "train.loss": last_loss,
                                     "train.ips": args.batch_size / iter_seconds},
                               verbosity=Verbosity.DEFAULT)
                    logger.log(step=(epoch, global_step),
                               data={"rank": 0, "scope": "aggregate",
                                     "bs": args.batch_size * world,
                                     "train.loss": last_loss,
                                     "train.ips": (args.batch_size * world) / iter_seconds},
                               verbosity=Verbosity.DEFAULT)
                    log.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps(row, allow_nan=False), flush=True)
                if args.checkpoint_every and global_step % args.checkpoint_every == 0:
                    save_checkpoint(output / "checkpoint.pth", raw_model, optimizer, scheduler,
                                    args, epoch, cursor, global_step, assets, rank, world,
                                    "step%d" % global_step)
                # --max-seconds 是墙钟量，各 rank 可能略有先后：统一裁决后再退出。
                stop = broadcast_decision(
                    bool((args.max_train_steps and invocation_steps >= args.max_train_steps)
                         or (args.max_seconds and time.monotonic() - started >= args.max_seconds)),
                    device, world)
                if stop:
                    break
            if cursor == epoch_length:
                scheduler.step()
                completed_epochs.append(epoch)
                epoch += 1
                cursor = 0
                save_checkpoint(output / "checkpoint.pth", raw_model, optimizer, scheduler,
                                args, epoch, cursor, global_step, assets, rank, world,
                                "epoch%d" % epoch)
                if not stop:
                    last_evaluation = evaluate_model(
                        raw_model, criterion, postprocessors, dataset_val, args, device,
                        output, logger, "evaluation-epoch-%02d.json" % epoch, rank, world)
    training_seconds = time.monotonic() - started
    if not args.eval:
        save_checkpoint(output / "checkpoint.pth", raw_model, optimizer, scheduler,
                        args, epoch, cursor, global_step, assets, rank, world, "final")
    # A bounded run evaluates the requested subset explicitly. Only the default
    # full evaluator may produce a competition AP result.
    if epoch == 20 and last_evaluation is not None:
        evaluation = last_evaluation
        if rank == 0:
            atomic_json(output / "evaluation.json", evaluation)
    else:
        evaluation = evaluate_model(raw_model, criterion, postprocessors, dataset_val, args,
                                    device, output, logger, "evaluation.json", rank, world)
    result = {"status": "evaluation_complete" if args.eval else
              "full_20_epochs_complete" if epoch == 20 else "bounded_training_checkpoint",
              "epoch": epoch, "batch_cursor": cursor, "global_step": global_step,
              "invocation_steps": invocation_steps, "completed_epochs": completed_epochs,
              "training_seconds": training_seconds, "last_loss": last_loss,
              "evaluation": evaluation, "native_calls": counts,
              "rank": rank, "world_size": world, "device": str(device),
              "peak_allocated_mib": torch.sdaa.max_memory_allocated() / (1024 * 1024),
              "peak_reserved_mib": torch.sdaa.max_memory_reserved() / (1024 * 1024),
              "checkpoint": None if args.eval else str(output / "checkpoint.pth"),
              "limits": ["native FP32 eager first-order MSDA with real DDP=%d over the "
                         "four core groups (TCCL all-reduce); no AMP" % world,
                         "official CPU Hungarian matching is retained",
                         "subset AP is diagnostic and does not meet the full test gate"]}
    if rank == 0:
        atomic_json(output / "result.json", result)
        logger.flush()
        print(json.dumps({"result": result}, allow_nan=False), flush=True)


def install_common_coco_eval(coco_evaluator, iou_types):
    """把 rank0 上合并好的预测对应的 COCOeval 结果装进 coco_eval 对象.

    等价于 datasets.coco_eval.create_common_coco_eval，但**不做** rank 间
    all_gather：预测已经由 evaluate_distributed 收集到 rank0；原 merge() 里的
    all_gather 走 util.misc.all_gather（在 "cuda" 上建张量），在 SDAA/DDP 下
    既无意义也会直接抛 "Torch not compiled with CUDA enabled"。
    """
    for iou_type in iou_types:
        coco_eval = coco_evaluator.coco_eval[iou_type]
        eval_imgs = np.concatenate(coco_evaluator.eval_imgs[iou_type], 2)
        img_ids = list(np.unique(np.array(coco_evaluator.img_ids)))
        coco_eval.evalImgs = list(eval_imgs.flatten())
        coco_eval.params.imgIds = img_ids
        coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


@torch.no_grad()
def evaluate_distributed(raw_model, criterion, postprocessors, data_loader, base_ds, device,
                         output_dir, rank, world, tag):
    """分布式评估：每个 rank 跑自己的 shard，rank0 在合并后的同一份预测集合上做 COCOeval.

    只把预测结果 gather 到 rank0（不按 rank 各自算 AP），因此 rank0 的 AP 与单进程
    全量评估是同一份预测上的同一次 COCOeval，不引入按 rank 拆分的度量差异。
    """
    raw_model.eval()
    criterion.eval()
    iou_types = tuple(k for k in ("segm", "bbox") if k in postprocessors.keys())
    started = time.monotonic()
    loss_sums = {}
    count = 0
    results = {}
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = raw_model(samples)
        loss_dict = criterion(outputs, targets)
        batch = len(targets)
        for key, value in loss_dict.items():
            loss_sums[key] = loss_sums.get(key, 0.0) + float(value.detach().cpu()) * batch
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        batch_results = postprocessors["bbox"](outputs, orig_target_sizes)
        for target, result in zip(targets, batch_results):
            image_id = int(target["image_id"].item())
            results[image_id] = {k: v.detach().cpu() for k, v in result.items()}
        count += batch
    payload = {"results": results, "loss_sums": loss_sums, "count": count}
    gathered = gather_payloads(payload, Path(output_dir), "eval-" + tag, rank, world)
    if rank != 0:
        return None, None
    merged_results = {}
    merged_sums = {}
    total = 0
    for item in gathered:
        merged_results.update(item["results"])
        total += item["count"]
        for key, value in item["loss_sums"].items():
            merged_sums[key] = merged_sums.get(key, 0.0) + value
    coco_evaluator = CocoEvaluator(base_ds, iou_types)
    coco_evaluator.update(merged_results)
    install_common_coco_eval(coco_evaluator, iou_types)
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    stats = {key: value / total for key, value in merged_sums.items()}
    # 与 engine.evaluate 的 train/val 口径一致：总 loss = 各个已加权分项之和的逐图均值
    weight_dict = criterion.weight_dict
    stats["loss"] = sum(value for key, value in merged_sums.items()
                        if key in weight_dict) / total
    stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()
    print("Distributed evaluation: images=%d ranks=%d wall=%.3fs"
          % (total, world, time.monotonic() - started), flush=True)
    return stats, coco_evaluator


def evaluate_model(raw_model, criterion, postprocessors, dataset, args, device, output, logger,
                   name, rank, world):
    count = args.eval_images or len(dataset)
    if not 0 < count <= len(dataset):
        raise ValueError("invalid --eval-images")
    if world > 1:
        # 为什么要求整除：SetCriterion.forward 内部会对 num_boxes 做
        # all_reduce（deformable_detr.py），所以每个 rank 必须跑**同样多**的
        # criterion 调用；空 shard 或长度不一的 shard 会让其它 rank 永久阻塞
        # 在这次 all_reduce 上（实测 2 张图 / 4 rank 直接挂死）。
        # 4952 张全量评估满足 4952 % (4*2) == 0。
        divisor = world * args.batch_size
        if count % divisor:
            raise ValueError(
                "--eval-images %d is not divisible by world_size*batchsize = %d*%d = %d; the "
                "distributed evaluator needs the same number of loss batches on every rank "
                "(SetCriterion all-reduces num_boxes). Use --eval-images 8 or 4952, or "
                "--nproc-per-node 1." % (count, world, args.batch_size, divisor))
        chunk = count // world
        start = rank * chunk
        indices = range(start, start + chunk)
    else:
        indices = range(count)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices),
        batch_size=args.batch_size, shuffle=False, collate_fn=utils.collate_fn,
        num_workers=0, pin_memory=False,
        generator=torch.Generator().manual_seed(args.seed))
    saved_rng = rng_state()
    started = time.monotonic()
    try:
        if world > 1:
            stats, _ = evaluate_distributed(raw_model, criterion, postprocessors, loader,
                                            get_coco_api_from_dataset(dataset), device,
                                            str(output), rank, world, name)
        else:
            stats, _ = evaluate(raw_model, criterion, postprocessors, loader,
                                get_coco_api_from_dataset(dataset), device, str(output))
        torch.sdaa.synchronize()
    finally:
        # Validation resize still consumes Python RNG. Keep evaluation from changing
        # the subsequent training augmentation sequence.
        restore_rng(saved_rng)
    elapsed = max(time.monotonic() - started, 1e-9)
    if rank != 0:
        return None
    logger.log(step="evaluation", data={"rank": 0, "val.loss": stats["loss"],
                                        "val.ips": count / (elapsed * world)},
               verbosity=Verbosity.DEFAULT)
    logger.log(step="evaluation", data={"rank": 0, "scope": "aggregate",
                                        "val.loss": stats["loss"],
                                        "val.ips": count / elapsed},
               verbosity=Verbosity.DEFAULT)
    result = {"images": count, "full_test": count == 4952, "stats": stats,
              "world_size": world, "wall_seconds": elapsed}
    atomic_json(output / name, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__, parents=[get_args_parser()])
    # lr/lr_backbone 默认随 world size 线性放大：单核组配方是 2e-5/2e-6，
    # DDP4 的全局 batch 是 4 倍，因此默认 8e-5/8e-6（显式 --lr 仍然优先）。
    parser.set_defaults(device="sdaa:0", epochs=20, num_feature_levels=1, num_classes=21,
                        lr=None, lr_backbone=None, lr_drop=14, num_workers=0,
                        coco_path="./data/voc_coco")
    parser.add_argument("--pretrained", default="./r50_deformable_detr_single_scale-checkpoint.pth")
    parser.add_argument("--max-train-steps", type=int, default=0,
                        help="bound this invocation; zero runs to the total 20-epoch target")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="stop after a completed minibatch; zero has no time bound")
    parser.add_argument("--eval-images", type=int, default=0,
                        help="diagnostic prefix; zero evaluates all 4952 test images")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--nproc-per-node", type=int, default=1,
                        help="1 = one core group (single process), 4 = the card's four core "
                             "groups under torchrun; run_scripts/test.sh defaults to 4")
    parser.add_argument("--audit-msda", action="store_true")
    parser.add_argument("--no-audit-msda", dest="audit_msda", action="store_false",
                        help="disable the native MSDA fwd/bwd call counter "
                             "(--audit-msda sets it back on; the entry defaults it on)")
    main(parser.parse_args())
