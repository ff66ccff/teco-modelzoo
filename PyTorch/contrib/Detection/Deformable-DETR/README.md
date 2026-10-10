# Deformable-DETR VOC2007 SDAA training

本目录提供赛题 5 的 ResNet-50 single-scale 训练、COCO AP 评估和断点续训入口。源码固定为 [Tecorigin/Deformable-DETR @ 1bda1f9](https://github.com/Tecorigin/Deformable-DETR/tree/1bda1f956a326d2601302413e4618c0fd109e799)，保留原网络、损失、增强和 Apache 版权声明；来源及文件哈希见 [SOURCE.json](SOURCE.json)。MSDeformAttn 在导入时绑定到 [Teco-Ops PR #40](https://github.com/Tecorigin/teco-ops/pull/40) 的配对原生前向和一阶反向。模型路径不使用 CPU grid_sample；CPU Hungarian matching 按官方实现保留。

本队厂商 py312 环境已完成 20 轮训练及完整 4,952 张 VOC2007 test 集评估，COCO AP@[.50:.95]=0.6013724164870897，超过本次 0.570 门槛。官方 ModelZoo Python 3.11 环境和 maintainer CI 仍未验证，因此不宣称适配完成。对象检测没有 32-token 生成接口；对应 smoke 为真实 VOC 前向、criterion、反向、优化器更新与检测评估。

## Environment

使用已有厂商环境，固定 `/home/py312/bin/python`，解析路径为 `/usr/local/python/bin/python3.12`。本轮验证为 Python 3.12.13、PyTorch 2.12.0a0+git0d62256、Torch-SDAA 20260623.8.51.dev0+gitd942f23、driver/runtime 3.2.0，单个 15 GiB 逻辑 SDAA 设备。不升级或改动全局 PyTorch、Torch-SDAA、torchvision 等耦合依赖。非耦合依赖列于 [requirements.txt](requirements.txt)；TCAP logger 固定源码在 `tcap_dllogger/`，无需全局安装。

交付入口用**解释器能力契约**取代了历史上的 py312 路径硬断言（见 [runtime_contract.py](runtime_contract.py) 与
[docs/bootstrap-py311.md](docs/bootstrap-py311.md)）：`PYTHON` 可指向官方 Python 3.11 或厂商 3.12（默认仍是
`/home/py312/bin/python`），入口再校验真实能力——Python 主次版本、`torch`/`torchvision`、`torch_sdaa` 且
`torch.sdaa.is_available()`、以及**已安装的 `tecoops` whl**（`import tecoops`；缺包即 fail-closed，不存在 torch 回退）。`TECOOPS_API_ROOT` 不再是必需项，仅作为源码树开发态的可选覆盖保留。
任一项不满足即 fail-closed 并打印可执行提示；**不允许 CPU fallback**。官方 py3.11 会话先用
`bash run_scripts/precheck_official_env.sh` 做静态预检（只导入与哈希，不装包、不占卡）。

官方 [ModelZoo 适配指南](https://github.com/Tecorigin/teco-modelzoo/blob/45e6b89185c8e6e081ce58d759cd62b8d15a1f5c/PyTorch/doc/模型适配指南.md)指定 Python 3.11 / PyTorch 2.7.1，与本队 AGENTS.md 的 py312 硬约束不同。本提交只证明上述 py312 栈；官方 py311 栈尚未执行；maintainer CI 也未验证。Full20 结果仅说明本队 py312 环境的全测试门槛通过，不表示官方环境门禁或适配完成。

原生 MSDA 通过**已安装的 `tecoops` whl** 绑定：`pip install tecoops-<version>-cp3xx-cp3xx-linux_loongarch64.whl` 后 `import tecoops`，模型调用 `tecoops.ms_deform_attn`（前向）与配对的一阶反向；绑定点是 `models/ops/_tecoops_binding.py`，在**导入期一次绑定**（热路径零分支、零 `getenv`），缺包 / 缺原生对象 / 缺入口一律 fail-closed，**不依赖源码树路径**，也不要求 `TECOOPS_API_ROOT`（该变量仅作为源码树开发态的可选覆盖：设置时 `tecoops` 必须解析在其之下）。

本轮实测 whl 由 PR #40 head `ff17c0ed940d810ccda0cdff965a7b54d0b06877` 的源码在隔离副本构建（`WITH_TORCH=ON WITH_INFERENCE_PLUGIN=OFF /home/py312/bin/python setup.py bdist_wheel`，cp312 / linux_loongarch64）：文件名 `tecoops-0.0.0-cp312-cp312-linux_loongarch64.whl`，**SHA-256 `e16d2f4b8555c640b9fcae09f7c4c5050648a1aeaf72f541edf29f049194f068`**（3,276,490 字节）；包内 `tecoops/_torch_ext.cpython-312-loongarch64-linux-gnu.so` = `e0424d1e…`、`tecoops/libteco_ops.so` = `6bd4ed1a…`。这四个算子源码（`.scpp`/`.h`/`find_*.cpp`/`.hpp`）与本交付树 api 构建的源码 SHA-256 **4/4 相同**；whl 内的 `.so` 与交付树 api 构建（`5e244917…`/`d3ae7b74…`）**字节不同**——同源码在不同 build root 下会差字节，本仓库对此的口径是**只做同 build 比较**。whl **不随 git 提交**；构建命令、源码 SHA 与获取方式见 [docs/bootstrap-py311.md](docs/bootstrap-py311.md)。

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
# paired native MSDA: install the wheel for the running interpreter (required)
pip install /path/to/tecoops-<version>-cp3xx-cp3xx-linux_loongarch64.whl
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

## Verification, 2026-10-06 and 2026-10-07

| Gate | Result |
| --- | --- |
| Actual VOC IDs 1/2, L1 model-boundary CPU oracle | 12 native model calls, 286 finite parameter gradients; forward max error 2.503395e-6, gradient max error 3.099442e-6; original atol/rtol preserved |
| Checkpoint restore and continued updates | 549 model tensors, 858 optimizer tensors, 3 RNG tensors restored exactly; scheduler/scalars/cursor matched; trained classifier retained, optimizer step 2→4 |
| Real changing-batch steady training | 341 optimizer updates / 330.203398 s; each batch native forward/backward 12/12; peak allocated 2771.174 MiB, reserved 4304 MiB |
| Evaluation plumbing | 32 real test images remain diagnostic; the full 4,952-image result is recorded separately below |
| Syntax, clean source wrapper, repository hooks | 43 source files verified, Python syntax and git diff --check pass; four unmodified official hooks pass; public wrapper resumed step 345→346 and evaluated 32 images |
| Full 20 epochs / full-test AP >= 0.570 | Passed on py312: 20/20 epochs; global step 50,100 (49,754 this invocation); full_test=true on 4,952 images; COCO AP@[.50:.95] 0.6013724164870897. All 49,754 logged loss and grad_norm values finite; 1,191,288 forward / 597,048 backward calls; peak allocated/reserved 2771.765625/5862 MiB. No performance improvement claimed. |


Full20 的原始 result.json 明确记录 full_test=true 和 4,952 张图；其中 coco_eval_bbox[0] 为 0.6013724164870897，满足本次完整测试集 0.570 门槛。保留原始 result.json limits 中通用的 subset AP 提示不变；该提示描述子集诊断，full20 记录里的 full_test 标记和图像数说明这次评估覆盖完整测试集。训练循环 steps.jsonl 共 49,754 条记录，逐条 loss 与 grad_norm 均为有限值。

full20 的 startup.json 记录 trainval/test 图像数 5,011/4,952，训练集 JSON、测试集 JSON 和官方预训练权重 SHA-256 分别为 e511d3a0e39f45162887e4d9f0bed63bf8f01e4c421d965445185b72990f1ab5、14edfb83a2525773c9e5b6e6d4bfa57a866de1cb3ca164c46afe0ac97629b561 和 d442fb2365d6e9640347b2b38686089d13cd55a5c3790d63e3602d079791fed1，与上方只读资产表一致。完成记录、原始 result/startup JSON 与 steps.jsonl 的路径和 SHA-256 见 validation.json。

训练使用代码 commit e8d314544ad96a27734172da0dc7382bcb87c6d7；完成时模型分支 HEAD 为 9318608eadc40e5ece26e5ac33a6c26f48b0131e。train_sdaa.py 和 SOURCE.json 的 SHA-256 与启动记录一致；训练入口、run_scripts 和 vendor 源码在该代码 commit 与分支 HEAD 之间无差异。此次仅更新文档，不改变训练入口或运行时代码。
原生一阶反向的列表归约顺序不保证位级确定性。连续训练与续训轨迹的首轮参数比较超过 atol=2e-5/rtol=1e-4，失败记录保留，未放宽该门限；两次完全连续训练的最大参数漂移为 7.626414e-5，连续与续训为 7.621944e-5。检查点恢复起点本身完全相同；不宣称后续训练轨迹位级相同。全量 AP 结果见上表；官方 py311 与 maintainer CI 尚未验证，因此不据此宣称最终竞赛适配完成。

CPU-info 架构提示和官方 torchvision/meshgrid/SyntaxWarning 保留在原始 stderr。本入口仅验证单设备 FP32 eager first-order，未验证 AMP、DDP、二阶梯度或 torch.compile。历史 L4 随机权重性能结果不作为本 VOC L1 配置的证据。

参考正确性入口（与交付入口同一解释器、SDAA 运行库与**同一份已安装 whl**；`SDAA_VISIBLE_DEVICES` 固定单设备）：

```bash
bash run_scripts/verify_native_voc.sh --output /dev/shm/voc-native-focused.json
```

该包装脚本与 `run_scripts/test.sh` 共用 `run_scripts/_entry_env.sh`：`PYTHON` 默认厂商路径、`MODEL_ROOT` fail-closed、
在运行前跑同一能力契约；`TECOOPS_API_ROOT` **不再必需**（仅源码树开发态可选覆盖，设置时校验 `tecoops` 解析在其之下）。
`verify_native_voc.py` 会打印实际加载的 `tecoops.__file__` 与其原生对象 SHA-256 自证，避免审计发现的「文档入口静默验证到无关旧 wheel」。

交付训练入口是 `run_scripts/test.sh`（内部 `run_DeformableDETR.py` → `train_sdaa.py`）。同目录的 `run_deformable_detr.py` 是上游遗留脚本（调用原始 `main.py`），**不是**本交付的 SDAA 入口，保留仅为来源完整性。

## Producer grad_output 单机制验证，2026-10-07

本轮仅修改原生 List backward 的 producer grad_output 搬运：每 owner 采用64B对齐的 `T go_record[128]` 与精确 `D*sizeof(T)` blocking memcpy，再使用原 `load_scalar`。CAS/node/list ownership、corner value/consumer go DMA、几何、数学顺序、FP16 bit conversion、ABI和分核不变。模型训练入口及初始化绑定方式不变。

对应算子源固定为 [Teco-Ops PR #40](https://github.com/Tecorigin/teco-ops/pull/40) commit `de4b69dfc1b3b984957e26d8aa7033c03fd7db4b`；目标 kernel SHA-256 为 `17e2b03a8e4168861ee0d4bea4b62b9d39f84d38421e158b9152babd7b9ed23e`。独立构建测试的 candidate core SHA-256 为 `618195bf59c3bc5946d7865a4eddd6131eec93b788884b8b5a9d99e7656f9a56`，extension仍为上方 `4c44e07c…`。最终PR源码的18个runtime文件已按canonical Git字节与候选验证源对应；没有声称从该最终PR head重新构建wheel或官方CI结果。

原 backward132、offset216 exact、List48及forward全部通过。首次forward因旧bootstrap优先选installed package报AttributeError；原失败保留，显式startup选择私有api后同测试与原阈值通过。真实VOC L1（N2/S494/H8/D32/P4，encoderQ494/decoderQ300）FP16 micro三次候选均快于三次baseline，encoder/decoder median延迟减少6.09%/2.97%；FP32 micro区间重叠。

实际VOC两图FP32无capture/hooks计时：baseline `0.606179200/0.624600624/0.621106923 s`（median `0.621106923`），candidate `0.610509759/0.611365727/0.621194002 s`（median `0.611365727`）。1.568% median差小于baseline spread2.966%，三区间重叠，因此不声称稳定模型提速。两方peak allocated/reserved均为 `888.853516/958 MiB`。独立baseline/candidate correctness与候选 `380轮/300.711627 s` steady每轮原CPU oracle门限通过，12forward+12backward、286有限参数梯度；无optimizer update或checkpoint写入。上方Full20/AP只属于既有baseline，不作为本候选精度或性能证据。

完整原值/中位数、shape、来源映射、core/kernel/ext/image/checkpoint SHA、原阈值与限制见 [单机制公开证明](validation-producer-go-20261007.json)。本模型独立接入与验证对应 [ModelZoo PR #6](https://github.com/Tecorigin/teco-modelzoo/pull/6)。原始日志与过程任务保留在模型分支 `logs/deformable-detr/producer-go-record-20261007/` 和 `op_learning/attention/deformable-producer-go-record-20261007/`；历史L4 profile及其他模型结果不计入本轮证据。

## 免责声明
ModelZoo仅提供公共数据集的下载链接。这些公共数据集不属于ModelZoo, ModelZoo也不对其质量或维护负责。请确保您具有这些数据集的使用许可。基于这些数据集的模型仅可用于非商业研究和教育。

致数据集所有者：

如果您不希望您的数据集公布在ModelZoo上或希望更新ModelZoo中属于您的数据集，请在Github/Gitee中提交issue,我们将根据您的issue删除或更新您的数据集。衷心感谢您对我们社区的理解和贡献。
