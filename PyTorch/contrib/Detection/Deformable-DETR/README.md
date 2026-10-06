# Deformable-DETR VOC2007 SDAA training

本目录提供赛题 5 的 ResNet-50 single-scale 训练、COCO AP 评估和断点续训入口。源码固定为 [Tecorigin/Deformable-DETR @ 1bda1f9](https://github.com/Tecorigin/Deformable-DETR/tree/1bda1f956a326d2601302413e4618c0fd109e799)，保留原网络、损失、增强和 Apache 版权声明；来源及文件哈希见 [vendor/SOURCE.json](vendor/SOURCE.json)。MSDeformAttn 在导入时绑定到 [Teco-Ops PR #40](https://github.com/Tecorigin/teco-ops/pull/40) 的配对原生前向和一阶反向。模型路径不使用 CPU grid_sample；CPU Hungarian matching 按官方实现保留。

当前已通过有界训练、原生梯度参考、检查点恢复和评估入口验证。完整 20 轮及完整 4,952 张测试图 AP 仍待完成，不能标注为精度达标或比赛适配完成。对象检测没有 32-token 生成接口；对应 smoke 为真实 VOC 前向、criterion、反向、优化器更新与检测评估。

## Environment

使用已有厂商环境，固定 `/home/py312/bin/python`，解析路径为 `/usr/local/python/bin/python3.12`。本轮验证为 Python 3.12.13、PyTorch 2.12.0a0+git0d62256、Torch-SDAA 20260623.8.51.dev0+gitd942f23、driver/runtime 3.2.0，单个 15 GiB 逻辑 SDAA 设备。不升级或改动全局 PyTorch、Torch-SDAA、torchvision 等耦合依赖。非耦合依赖列于 [requirements.txt](requirements.txt)；TCAP logger 固定源码在 `vendor/tcap_dllogger/`，无需全局安装。

官方 [ModelZoo 适配指南](https://github.com/Tecorigin/teco-modelzoo/blob/45e6b89185c8e6e081ce58d759cd62b8d15a1f5c/PyTorch/doc/模型适配指南.md)指定 Python 3.11 / PyTorch 2.7.1，与本队 AGENTS.md 的 py312 硬约束不同。本提交只证明上述 py312 栈；py311 栈未执行，需单独验证，不能据此声称满足官方环境门禁。

`TECOOPS_API_ROOT` 指向项目独立构建/解包的 `tecoops` 包的父目录，须提供 `ms_deform_attn` 前向及一阶反向。运行时记录实际 extension/core 映射与 SHA-256；本轮验证的 extension 为 `4c44e07c3e02f9433416d2afe7c28d55b3b87c6bbac6f40f4b62549af240392a`，core 为 `ef437793422d1c7b89e4f1680b8430f53c5d28b7b21f3bae908911457853217d`。该构建的 18 个运行源码文件与 PR #40 head `b565d40242473c5a002886ae372944ba2d9070f3` 等价；本轮没有重新构建该 head。

## Read-only assets

`MODEL_ROOT` 引用已预置的官方 `data_ckpt` 目录，内含 `voc_coco/` 和 `r50_deformable_detr_single_scale-checkpoint.pth`。不下载第二份数据或权重。启动脚本只创建符号链接：数据位置为模型根目录 `./data/voc_coco`，预训练权重链接位于模型根目录，满足官方布局。

| Asset | Count / bytes | SHA-256 |
| --- | --- | --- |
| VOC2007 trainval JSON | 5,011 images / 15,662 annotations | e511d3a0e39f45162887e4d9f0bed63bf8f01e4c421d965445185b72990f1ab5 |
| VOC2007 test JSON (val directory) | 4,952 images / 14,976 annotations | 14edfb83a2525773c9e5b6e6d4bfa57a866de1cb3ca164c46afe0ac97629b561 |
| Official pretrained checkpoint | 407,659,336 bytes | d442fb2365d6e9640347b2b38686089d13cd55a5c3790d63e3602d079791fed1 |

类别 ID 为 1–20，分类头输出 21 通道。资产只读；训练产生的新检查点写入显式指定的输出目录，至少需要 2 GiB 空闲空间。每次调用使用一个新的空输出目录，避免覆盖失败现场。

## Training and resume

从干净检出按下面命令启动。环境变量替换为本机已有资产和原生包位置；没有权重复制、全局安装或修改框架步骤。

```bash
export MODEL_ROOT=/path/to/provisioned/data_ckpt
export TECOOPS_API_ROOT=/path/to/project-local/native/api
export SDAA_VISIBLE_DEVICES=0
cd run_scripts
bash test.sh --output-dir /dev/shm/voc20-run --epoch 20 --batchsize 2 --seed 42
```

默认总目标为 20 轮、FP32、R50、1 level、300 queries、6+6 layers、H8/D32/P4；AdamW lr=2e-5、backbone lr=2e-6、projection multiplier=.1、weight decay=1e-4、gradient clip=.1、StepLR drop=14。单进程同步 SDAA 搬运；`num_workers=0` 使增强 RNG 可恢复。每轮使用独立 `seed+epoch` 采样 generator，保存批次游标；batch=2 时每轮 2,505 批，沿用官方 `drop_last=True`。

预训练加载只重新初始化 COCO 91→VOC 21 类分类头。`--resume` 专用于本入口产生的训练检查点，严格恢复已训练分类头、三组 AdamW 状态、scheduler、Python/NumPy/Torch/SDAA RNG、epoch 和 batch cursor；恢复起点逐项精确检查。检查点以 `weights_only=True` 读取，初始官方 legacy 文件仅额外允许 `argparse.Namespace`。

```bash
# 继续同一总计 20 轮目标，不重新初始化分类头；使用新的输出目录。
bash test.sh --output-dir /dev/shm/voc20-resume \
  --resume /dev/shm/voc20-run/checkpoint.pth

# 完整测试集评估必须使用已训练检查点。
bash test.sh --output-dir /dev/shm/voc-eval \
  --resume /dev/shm/voc20-resume/checkpoint.pth --eval

# 有界集成检查；不代表完整训练或官方 AP。
bash test.sh --output-dir /dev/shm/voc-check \
  --max-seconds 330 --eval-images 32 --checkpoint-every 100
```

默认每 1,000 步及轮末/有界正常退出时原子覆盖一个 `checkpoint.pth`，不累计多份权重。输出包括 `startup.json`、逐步 `steps.jsonl`、TCAP `sdaa.log`（train.loss/train.ips/val.loss/val.ips，rank=0）、`evaluation.json` 和 `result.json`。运行脚本默认启用原生调用计数，每个训练批次必须有 12 次 native forward 与 12 次 native backward。

## Verification, 2026-10-06

| Gate | Result |
| --- | --- |
| Actual VOC IDs 1/2, L1 model-boundary CPU oracle | 12 native model calls, 286 finite parameter gradients; forward max error 2.503395e-6, gradient max error 3.099442e-6; original atol/rtol preserved |
| Checkpoint restore and continued updates | 549 model tensors, 858 optimizer tensors, 3 RNG tensors restored exactly; scheduler/scalars/cursor matched; trained classifier retained, optimizer step 2→4 |
| Real changing-batch steady training | 341 optimizer updates / 330.203398 s; each batch native forward/backward 12/12; peak allocated 2771.174 MiB, reserved 4304 MiB |
| Evaluation plumbing | 32 real test images, diagnostic AP only; full 4,952-image gate pending |
| Syntax, clean source wrapper, repository hooks | 43 source files verified, Python syntax and git diff --check pass; four unmodified official hooks pass; public wrapper resumed step 345→346 and evaluated 32 images |
| Full 20 epochs / official AP≥0.570 | Pending; no performance improvement claimed |

原生一阶反向的列表归约顺序不保证位级确定性。连续训练与续训轨迹的首轮参数比较超过 atol=2e-5/rtol=1e-4，失败记录保留，未放宽该门限；两次完全连续训练的最大参数漂移为 7.626414e-5，连续与续训为 7.621944e-5。检查点恢复起点本身完全相同；不宣称后续训练轨迹位级相同。完整 AP 与最终指标必须以全量结果判断。

CPU-info 架构提示和官方 torchvision/meshgrid/SyntaxWarning 保留在原始 stderr。本入口仅验证单设备 FP32 eager first-order，未验证 AMP、DDP、二阶梯度或 torch.compile。历史 L4 随机权重性能结果不作为本 VOC L1 配置的证据。

参考正确性入口（由正确厂商环境调用，SDAA_VISIBLE_DEVICES 固定单设备）：

```bash
cd ..
/home/py312/bin/python verify_native_voc.py --output /dev/shm/voc-native-focused.json
```

## 免责声明
ModelZoo仅提供公共数据集的下载链接。这些公共数据集不属于ModelZoo, ModelZoo也不对其质量或维护负责。请确保您具有这些数据集的使用许可。基于这些数据集的模型仅可用于非商业研究和教育。

致数据集所有者：

如果您不希望您的数据集公布在ModelZoo上或希望更新ModelZoo中属于您的数据集，请在Github/Gitee中提交issue,我们将根据您的issue删除或更新您的数据集。衷心感谢您对我们社区的理解和贡献。
