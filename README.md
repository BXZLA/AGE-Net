# AGE-Net

**Angle-gated crystal graphs for elastic-modulus prediction.**

从 CIF 结构预测体积模量 K 和剪切模量 G。模型、基础图卷积、角度特征与训练逻辑统一在 `agenet/` 中；包含完整固定划分、10,225 个 CIF 和 K/G 两套 seed-42 权重。

## 快速开始

在本目录执行。已运行环境为 Python 3.9.25、PyTorch 2.8.0、NumPy 1.26.4、pandas 2.3.3 和 pymatgen 2024.8.9，依赖集中在 `pyproject.toml`。推荐创建独立环境：

```bash
conda env create -f environment.yml
conda activate agenet-paper
python -m agenet.check
python -m agenet.predict --device cpu
```

已有 Python 3.9 环境时，可用 `python -m pip install -e .` 安装。预测默认读取 115 条 `test_B.csv`，输出 `outputs/predictions.csv`，同时给出 log10 和 GPa 数值。检查命令核对数据与权重，并执行两条结构的推理及一次不保存权重的反向传播。

## 训练

K 默认使用标量角度编码，G 默认使用 F=5 的 Fourier 角度编码：

```bash
python -m agenet.train --property K
python -m agenet.train --property G
```

可选参数包括 `--seed 42`、`--max-epochs 300`、`--device cpu` 和 `--run-name my_run`。每次运行保存到 `outputs/<run-name>/`，包含最优权重、标准化参数、训练日志及训练后预测。同名运行目录已存在时会停止，随附权重不会被训练覆盖。增加 `--report-tests` 才会在模型选择完成后输出测试集报告。

## 新材料预测

准备一份含唯一 `material_id` 的 CSV，以及同名 `<material_id>.cif` 文件：

```bash
python -m agenet.predict --input-csv new_materials.csv --cif-dir new_cifs --output outputs/new_predictions.csv
```

真值可省略；可选的 `K_VRH` 和 `G_VRH` 列需为正值，单位 GPa。默认自动选择可用的 GPU，`--device cpu` 可强制使用 CPU。

## 代码结构

```text
agenet/
    model.py       # AGENet、基础消息传递与角度门控
    data.py        # 周期图、近邻、键角与批处理
    angles.py      # 标量和 Fourier 角度编码
    train.py       # 混合训练池、早停与模型选择
    predict.py     # 权重加载和 K/G 推理
    check.py       # 数据、权重与运行检查
    config.py      # 路径和固定训练参数
    utils.py       # 标准化、随机种子和设备处理
data/
    atom_init.json
    splits/        # 固定划分 CSV 标签和 cif_files/
checkpoints/
    K/             # model_best.pth.tar、train_log.csv
    G/
outputs/
    test_B_predictions.csv  # 保留的历史参考预测
```

`agenet` 内部使用相对导入，无需单独安装 CGCNN。公共模型类为 `from agenet import AGENet`。本仓库按随附数据的工作目录使用，因此采用 editable 安装。

## 数据和设置

| 文件 | 数量 | 用途 |
|---|---:|---|
| `non_sse_train.csv` | 9,459 | 非 Li 源域训练 |
| `finetune_train_core.csv` | 536 | 含 Li 目标域训练 |
| `val_SSE.csv` | 115 | 早停和最优权重选择 |
| `test_B.csv` | 115 | 最终测试报告 |

四组按材料 ID 互不重叠，前两组合并为 9,995 条训练记录。标签为 Materials Project 派生的 VRH 模量，单位 GPa；历史 `is_sse` 列用作域标记。原始数据库查询脚本未包含在内，使用随附 CSV/CIF 无需数据库 API。

划分来源：含锂池由前序 `finetune_train.csv` 与旧 `test_B.csv` 合并后重新划分，旧测试集不再作为独立留出集；非含锂池合并自前序 `train.csv`、`val.csv` 和 `test_A.csv`。生成记录为 seed 20260512、验证/测试比例各 0.15，并从 50,000 个随机候选划分中进行平衡选择（balance_score = 1.375997）。本包直接复用随附的最终固定划分。

模型采用四层晶体图卷积，原子特征 64、隐藏特征 128，构图半径 8 Å、最多 12 个近邻、径向 Gaussian 步长 0.2 Å。K 将 θ/π 映射为 64 维角度特征；G 使用 F=5 的 Fourier 特征，计算见 `angles.py`。

训练目标为标准化的 log10(K) 或 log10(G)，标准化统计量从训练集中抽取最多 500 条标签计算。优化使用 MSE、Adam、学习率 0.001、batch size 64；第 150/220 epoch 将学习率乘以 0.1，最多训练 300 epoch，验证 MAE 连续 70 epoch 无改善时停止。随附最优 K/G 权重分别来自 epoch 138/153。

## 验证与来源

统一版按同一 CPU 环境与整合前实现对照，验证了图张量、115 条 K/G 预测、反向传播，以及两个实际训练入口的小样本单 epoch 结果。完整 300 epoch 和五种子论文实验未重新训练。

历史参考值另有一项已知边界：Windows CPU 的 14 个结构与历史结果超出舍入差异，最大 log10 差值为 K 0.009738、G 0.005159；两例代表结构已确认与第 12 个近邻处等距原子的选择有关。整合前后在相同环境下保持一致，原构图规则、数据和权重保留。

基础数据加载和晶体图卷积改编自 [CGCNN](https://github.com/txie-93/cgcnn)，现整合到 `data.py` 和 `model.py`，原始版权与 MIT 许可保留在 [LICENSE_CGCNN](LICENSE_CGCNN)。本次代码组织调整未指定 AGE-Net 自有代码或数据的整体许可证。
