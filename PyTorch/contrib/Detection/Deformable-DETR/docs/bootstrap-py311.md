# 官方 py3.11 / PyTorch 2.7.1 环境引导与预检（Deformable-DETR，赛题 5）

本文件是**静态引导件**：它把官方训练环境的准备步骤、离线依赖替代、预检命令和训练仓目标目录映射固化在树内。
本轮**没有**执行任何 py3.11 环境准备或设备运行（见文末「未执行项」）。

> 结论口径：`README.md` 里记录的全部精度/性能证据来自**本队厂商 py312 + Torch-SDAA** 环境
> （20 轮训练、4,952 张 VOC2007 test、COCO AP@[.50:.95]=0.6013724164870897）。官方 ModelZoo py3.11 / PyTorch 2.7.1
> 环境与 maintainer CI **仍未验证**，因此不宣称适配完成。本文件的作用是让那次验证只花一次 GPU 窗口。

---

## 1. 解释器能力契约（不再写死 py312 路径）

历史版本在三个入口写死了「解释器路径必须等于 `/home/py312/bin/python`」，这在官方 py3.11 环境下**必然启动失败且无变量可绕**。
现在改为**能力契约**（`runtime_contract.py`）：

| 校验项 | 要求 | 失败提示 |
| --- | --- | --- |
| Python 版本 | 3.11（官方）或 3.12（厂商） | 给出两者及 `PYTHON=<解释器>` 的覆盖方式 |
| `torch` / `torchvision` | 可导入、torch ≥ 2.x | 提示不得覆盖官方/vendor 的 torch 栈 |
| `torch_sdaa` | 可导入且 `torch.sdaa.is_available()` 为真、设备数 ≥ 1 | 提示 `source /opt/tecoai/setvars.sh`；**不允许 CPU fallback** |
| `tecoops` | **已安装的 whl** 可导入，含 `ms_deform_attn` / `ms_deform_attn_forward` / `ms_deform_attn_backward` 与 `_torch_ext*.so` | 提示 `pip install tecoops-<version>-cp3xx-…whl`；**不依赖源码树路径** |
| `tecoops` 来源 | 默认取**已安装包**；仅当设置了 `TECOOPS_API_ROOT`（源码树开发态可选覆盖）时，`tecoops.__file__` 必须**位于 root 之下** | 明确报出实际路径与 shadowing 原因，fail-closed；缺配对入口时同样 fail-closed（本机 vendor site-packages 就存在一个缺入口的旧 `tecoops`，契约会把它拦下）|

三个入口都调用同一契约：`run_scripts/test.sh`（shell 前置）、`run_scripts/run_DeformableDETR.py`、`train_sdaa.py`
（导入期执行，因此 `--help` 也走门禁）、`verify_native_voc.py`。解释器可用环境变量选择：

```bash
export PYTHON=/path/to/python3.11      # 官方环境
export PYTHON=/home/py312/bin/python    # 本队厂商环境（默认值）
```

---

## 2. 官方环境准备步骤

1. **进入官方 ModelZoo 训练镜像**：Python 3.11 + PyTorch 2.7.1 + torchvision 由镜像提供。
   **不要**用 pip 安装/升级/卸载 torch、torchvision；`requirements.txt` 也刻意不包含它们。
2. **安装配对的原生 `tecoops` whl（必需）**：从 Teco-Ops PR #40 的源码用该解释器构建原生前向 +
   一阶反向扩展并打包：
   ```bash
   # 在 Teco-Ops PR #40 的 checkout 内（head ff17c0e 侧）
   WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF "$PYTHON" setup.py bdist_wheel
   "$PYTHON" -m pip install dist/tecoops-<version>-cp3xx-cp3xx-linux_loongarch64.whl
   ```
   模型侧通过 `models/ops/_tecoops_binding.py` 在**导入期一次绑定** `tecoops`；缺包即 fail-closed，
   不会回退到 torch 实现。`TECOOPS_API_ROOT` **不再必需**，仅作为源码树开发态的可选覆盖
   （设置时入口会把它前插到 `PYTHONPATH` 并校验 `tecoops` 解析在其之下）。
3. **非耦合依赖**（`requirements.txt`，已钉版）：
   ```bash
   "$PYTHON" -m pip install --no-index --find-links /path/to/wheels -r requirements.txt
   ```
   离线时使用本地 wheel 目录；**不需要**联网拉取 `git+https://…tcap_dllogger.git@6997d7d…`，
   因为该源码已随树 vendor 在 `tcap_dllogger/`（来源与哈希见其 `SOURCE.json`），
   `train_sdaa.py` 通过 `sys.path` 直接引用 vendor 副本。
4. **只读资产**：`export MODEL_ROOT=/path/to/provisioned/data_ckpt`，其中含
   `voc_coco/train/train.json`、`voc_coco/val/val.json` 与 `r50_deformable_detr_single_scale-checkpoint.pth`；
   入口只创建符号链接，不复制权重。
5. **先跑预检，再花 GPU**：
   ```bash
   PYTHON=/path/to/python3.11 \
   MODEL_ROOT=/path/to/provisioned/data_ckpt \
   bash run_scripts/precheck_official_env.sh
   ```
   预检只做导入与哈希：Python 版本（默认要求 3.11，可用 `EXPECTED_PYTHON_MINOR` 改为 3.12 以检查厂商栈）、
   torch 版本前缀（默认 `2.7.1`，可用 `EXPECTED_TORCH_PREFIX` 覆盖）、SDAA 可用性、`tecoops` 实际加载路径与原生对象 SHA-256（防 shadowing）、
   三个资产 SHA-256。任意必需项失败即非零退出。

---

## 3. 官方环境下建议执行的门禁

```bash
# 3.1 有界集成检查（330 s / 32 图；不代表完整训练或官方 AP）
bash run_scripts/test.sh --output-dir /dev/shm/voc-check --max-seconds 330 \
    --eval-images 32 --checkpoint-every 100

# 3.2 参考正确性入口（真实 VOC 边界上的原生前向/梯度 oracle，打印 tecoops.__file__ 自证）
bash run_scripts/verify_native_voc.sh --output /dev/shm/voc-native-focused.json
```

需要重跑什么：① 有界集成检查；② 参考正确性入口（确认原生调用计数 12 forward + 12 backward/step 与 py312 证据一致）；
③ 至少一次评估路径连通。**20 轮全量训练与完整 test AP 是否必须在 py3.11 上重跑，属成本/收益决策**，
需在 GPU 窗口与队长确认后再执行。

---

## 4. 训练仓（teco-modelzoo）目标目录映射

赛题 5 属**小模型**，按 `docs/competition/requirements.md` 应提交到**模型训练仓库**（本目录没有 `model_adaptations/`）。

| 本仓库路径 | teco-modelzoo 目标路径（PR #6 实测） |
| --- | --- |
| `models/deformable-detr/` | `PyTorch/contrib/Detection/Deformable-DETR/` |
| `models/deformable-detr/run_scripts/` | `PyTorch/contrib/Detection/Deformable-DETR/run_scripts/` |
| `models/deformable-detr/vendor/` | `PyTorch/contrib/Detection/Deformable-DETR/`（**vendor/ 在目标仓已上提一层**，见该仓 `RELOCATION-NOTE.md`） |
| `custom_ops/deformable_detr/`（算子侧） | 自定义算子仓库 Teco-Ops（PR #40），不随训练仓提交 |

- 对应 PR：[Tecorigin/teco-modelzoo#6](https://github.com/Tecorigin/teco-modelzoo/pull/6)，
  head `d0f77f5a593be052b3b479da37702ba65fe966f6`，base `main` `45e6b89185c8e6e081ce58d759cd62b8d15a1f5c`，
  44 个新增文件，状态 `open`（待维护者批准）。本轮**未 push、未改动该 PR**。
- 上游约定出处：ModelZoo 适配指南
  `PyTorch/doc/模型适配指南.md`（base 同上）。
- **待人工确认（未在本文件断言）**：提交分支口径在三份文档间不一致——小模型文档示例 `op/ms_deform_attn` 与
  `contrib/Deformable-DETR`，FAQ 又说「最终以 main 分支为准」。确认前不把任一形态当作官方要求。

---

## 5. 未执行项与边界

- 本轮**未**准备 py3.11 环境、未安装依赖、未构建 py3.11 版 `tecoops`、未上卡、未运行训练/评估。
- 本文件只固化静态步骤；§3 的门禁需要 GPU 窗口。
- py3.12 的全部既有证据（`README.md` 验证表、`validation.json`、`validation-producer-go-20261007.json`）**不能**替代
  py3.11 复跑结论，两者需分别记录。
