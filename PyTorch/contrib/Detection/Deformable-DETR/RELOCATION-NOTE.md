# RELOCATION-NOTE.md — `vendor/` flattening: provenance impact

## What changed and why

To match the organizer's ModelZoo layout, this contribution's `vendor/` subtree was flattened one
level up: files that used to live under `vendor/` now sit directly in this directory
(`PyTorch/contrib/Detection/Deformable-DETR/`). A follow-up commit repairs the two path expressions
in `train_sdaa.py` that still assumed the old `vendor/` level, plus three stale path references in
`README.md` and one stale `.gitignore` rule.

| file | pre-fix sha256 | post-fix sha256 |
| --- | --- | --- |
| `train_sdaa.py` | `0378feb7fdeb155c10a1d7307a2f28882d1d1997939ad6810bfd7c504fead29a` | `d9458eb47c4da78ac66632d7ad98300a3d1417f06737c7838e8a74bdd3a89f84` |
| `README.md` | `3f6718dff76865bf351409c830c824848007492e62a9d7bd91154af8f013bbe6` | `029bd4d06cbea46b099514ad0d4cfcf30d3f14df6ce27ba01d2d079830593a97` |
| `.gitignore` | `d7da95b8b5eed9fae435540c51197ec554ee8acea979e5586026fa151d026935` | `e482f8651417526a8720ec66898a64bed9df30e5aef900e941b36dac44de3a9e` |
| `verify_native_voc.py` | `5ba671cdb4fdf5940845c6636925379c922a33fb93116f1d505d0dc5a9739d8f` | unchanged |
| `SOURCE.json` | `4bc34de500379e8da9d8116667b2bb6e0d302fed5e8e25f54f2f51d951d2280d` | unchanged |

## The delta in the training entry is two path expressions only

```diff
-sys.path.insert(0, str(HERE / "vendor"))
+sys.path.insert(0, str(HERE))
-               "vendor_manifest_sha256": sha256(HERE / "vendor/SOURCE.json"),
+               "vendor_manifest_sha256": sha256(HERE / "SOURCE.json"),
```

There is **no change to model, loss, augmentation, optimizer, scheduler, RNG, resume-check or
logging logic**. `sys.path.insert(0, str(HERE))` preserves the original intent: `HERE` is already
`sys.path[0]` when this entry point runs, and every module imported immediately after
(`main`, `models`, `datasets`, `engine`, `util.misc`, `tcap_dllogger`) lives in `HERE`. The receipt
key name `vendor_manifest_sha256` is deliberately left unchanged so downstream consumers keep
working.

**The recorded 20-epoch AP=0.6013724164870897 result is therefore unaffected.** The repaired entry
performs the same computation and only resolves the same two paths to their actual location.

## `validation.json` is a frozen historical record — do not edit it

`validation.json:246` (`entry_sha256`) and `:433` (`training_entry_sha256`) record the pre-fix
`0378feb7…` as the entry that actually produced AP=0.6014, and `:247` / `:434` record
`vendor_manifest_sha256 = 4bc34de5…`. Those remain true statements about the artifact that was
validated. `validation.json` must **not** be edited to match the new hash; the post-fix hash is
recorded at run time in `startup.json` (`entry_sha256`) instead.

`SOURCE.json` itself is unchanged by this fix, so the `vendor_manifest_sha256` value recorded in
`validation.json` is still exactly the hash of the delivered manifest — only its directory depth
changed. All 28 `delivered_sha256` entries in `SOURCE.json` and all 3 in
`tcap_dllogger/SOURCE.json` were re-verified against the delivered files: 0 missing, 0 mismatches.

## Could the relocation affect anything else?

Checked, and it does not, apart from one stale comment and the documented asset dependencies:

* Every path expression resolved relative to a file's own directory now exists. The only
  non-existent ones are deliberate `MODEL_ROOT` assets — `data/voc_coco` and
  `r50_deformable_detr_single_scale-checkpoint.pth` — which `run_scripts/test.sh:27-29` symlinks in
  from `$MODEL_ROOT` at run time. They are data, not code, and stay git-ignored.
* `train_sdaa.py:29` still reads `# Licensed under the Apache License, Version 2.0 (see vendor/LICENSE).`
  The referenced file is now `LICENSE` in this directory. This is a **comment only** with no runtime
  effect. It was left untouched on purpose, because the follow-up commit is scoped to the two
  runtime path expressions. **Known, recorded residue.**
* The three `vendor/...` references in `README.md` were updated to the flattened paths
  (`SOURCE.json`, `tcap_dllogger/`), and the provenance sentence at line 75 was qualified so that it
  no longer claims the entry hash still equals the frozen startup record.
* `.gitignore`'s stale `vendor/**/__pycache__/` rule was dropped; the preceding `__pycache__/` rule
  already covers every `__pycache__` directory, so this is behaviour-preserving.
* `validation.json:439` lists `models/deformable-detr/vendor/` inside the frozen historical
  `unchanged_source_paths` record. That describes the pre-relocation tree and is frozen with the rest
  of `validation.json`; it is not edited.
* No other file in the repository references the `vendor/` directory.

结论：本目录扁平化后，`train_sdaa.py` 的改动仅限两处路径表达式，不改变模型、损失、增强、优化器、
调度器、RNG、续训校验或日志逻辑，已记录的 AP=0.6014 结果不受影响；`validation.json` 为冻结历史
证据，不随之修改。
