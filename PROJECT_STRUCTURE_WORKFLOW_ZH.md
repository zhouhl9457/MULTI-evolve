# MULTI-evolve 项目结构、方法库与完整数据流说明

这份文档用自然语言解释文献对应代码 `VincentQTran/MULTI-evolve` 的整体结构、内置方法、实际运行链条、脚本之间的数据流，以及函数和类之间是如何互相引用的。它不是安装手册，而是帮助你读懂代码的“地图”。

## 1. 项目在解决什么问题

MULTI-evolve 的目标是把“少量突变实验数据”转化为“可实验构建的多突变体设计”。它的核心想法可以概括成一句话：

给定一个野生型蛋白序列和一批已测过活性的突变体，先训练机器学习模型预测突变体性能，再用模型筛选组合突变，最后把候选突变转换成实验克隆所需的寡核苷酸设计。

因此，项目不是只有一个模型文件，而是一整套流程：

1. 读入训练数据和野生型序列。
2. 把氨基酸序列编码成数值特征。
3. 划分训练、验证、测试数据。
4. 训练预测模型。
5. 比较不同模型/超参数的表现。
6. 用最佳模型提议组合突变体。
7. 导出候选突变 CSV。
8. 将候选突变转成 MULTI-assembly 寡核苷酸设计。

另外还有一条蛋白语言模型 zero-shot 支线：不用当前数据训练模型，而是直接用 ESM、ESM-IF 等预训练模型给所有单点突变打分，用于提名单点突变。

## 2. 项目目录的宏观分工

根目录中最重要的是这些部分：

| 路径 | 作用 |
|---|---|
| `scripts/` | 命令行工作流入口。论文主流程主要通过这里的脚本串起来。 |
| `app.py` | Streamlit 网页应用入口，把训练、提议突变、寡核苷酸设计、zero-shot 包装成网页界面。 |
| `multievolve/featurizers/` | 特征化器，把序列、突变或语言模型分数转换为模型可用的数值矩阵。 |
| `multievolve/predictors/` | 预测模型，包括传统回归模型、高斯过程、PyTorch 神经网络。 |
| `multievolve/proposers/` | 候选突变生成器，包括组合突变枚举、模型引导搜索、模拟退火等。 |
| `multievolve/splitters/` | 数据切分器，把实验数据划分成训练、验证、测试集合。 |
| `multievolve/utils/` | 工具层，包括突变格式转换、缓存、zero-shot 打分、寡核苷酸设计等。 |
| `data/` | 示例数据，包括示例蛋白、突变数据、突变池、结构文件。 |
| `notebooks/` | 教程和 benchmark 示例。 |

可以把整个项目想成一个“可插拔流水线”。`featurizers`、`splitters`、`predictors`、`proposers` 都提供了很多选择；论文命令行主流程实际选了其中一条较稳健、依赖较少的路径。

## 3. 项目内置的方法库

### 3.1 Featurizers：把蛋白序列变成模型输入

`featurizers` 是项目最重要的模块之一。它解决的问题是：机器学习模型不能直接读氨基酸字母，所以要先把序列或突变转成数值特征。

项目内置的 featurizer 大致分成五类。

第一类是基础编码：

| 方法 | 类/名称 | 含义 |
|---|---|---|
| one-hot | `OneHotFeaturizer` | 把每个氨基酸位置编码为 one-hot 向量。 |
| Georgiev | `GeorgievFeaturizer` | 用 Georgiev 氨基酸理化性质参数编码。 |
| AAIndex/PCA | `AAIdxFeaturizer` | 用 AAIndex 派生的氨基酸性质向量编码。 |

第二类是蛋白语言模型 embedding 或 logits：

| 方法 | 类/名称 | 含义 |
|---|---|---|
| ESM logits | `ESMLogitsFeaturizer` | 输出 ESM 对每个位点氨基酸的 log probability 类特征。 |
| ESM-1v embedding | `ESM1vEmbedFeaturizer` | 用 ESM-1v 得到序列表征。 |
| ESM-2 3B embedding | `ESM2EmbedFeaturizer` | 用 ESM-2 3B 模型得到 embedding。 |
| ESM-2 15B embedding | `ESM2_15b_EmbedFeaturizer` | 用更大的 ESM-2 15B 模型得到 embedding。 |
| ESM-C/Forge | `ForgeESMFeaturizer`、`Forge_ESMC_6B_EmbedFeaturizer` | 通过 Forge API 风格调用 ESM-C 模型。 |
| Ankh | `AnkhBaseFeaturizer`、`AnkhLargeFeaturizer` | 用 Ankh 蛋白语言模型生成 embedding。 |
| ProtT5 | `ProtT5_XL_U50_EmbedFeaturizer` | 用 ProtT5 生成蛋白序列表征。 |
| UniRep | `UnirepBaseFeaturizer`、`EvotunedUnirepFeaturizer` | 用 UniRep 或 evotuned UniRep 生成表征。 |

第三类是 MSA 相关方法：

| 方法 | 类/名称 | 含义 |
|---|---|---|
| MSA embedding | `MSAEmbedFeaturizer` | 用 MSA Transformer 输出 MSA embedding。 |
| MSA sequence embedding | `MSASequenceEmbedFeaturizer` | 输出按序列汇总的 MSA 表征。 |
| MSA logits | `MSALogitsFeaturizer` | 输出 MSA Transformer 的 logits 类特征。 |

第四类是 zero-shot 分数特征：

| 方法 | 类/名称 | 含义 |
|---|---|---|
| ESM zero-shot | `ZeroshotESMFeaturizer` | 把 ESM 对突变的 zero-shot 分数作为特征。 |
| MSA zero-shot | `ZeroshotMSAFeaturizer` | 把 MSA Transformer zero-shot 分数作为特征。 |
| CSCS | `ZeroshotCSCSFeaturizer` 等 | 使用 CSCS 相关 zero-shot 分数。 |
| ProSE | `ZeroshotProseFeaturizer` | 使用 ProSE 相关分数。 |
| ESM-IF | `ZeroshotESMIFFeaturizer` | 使用结构感知的 ESM-IF 分数。 |

第五类是组合特征：

| 方法 | 类/名称 | 含义 |
|---|---|---|
| one-hot + Georgiev | `OnehotAndGeorgievFeaturizer` | 拼接 one-hot 和 Georgiev 特征。 |
| one-hot + AAIndex | `OnehotAndAAIdxFeaturizer` | 拼接 one-hot 和 AAIndex 特征。 |
| one-hot + ESM logits | `OnehotAndESMLogitsFeaturizer` | 把 one-hot 与 ESM logits 堆叠。 |
| one-hot + MSA logits | `OnehotAndESMMSALogitsFeaturizer` | 把 one-hot 与 MSA logits 堆叠。 |
| one-hot + ESM2 embedding | `OnehotAndESM2EmbedFeaturizer`、`OnehotAndESM2_15bEmbedFeaturizer` | 把 one-hot 与 ESM2 embedding 结合。 |

这些可用选项集中登记在 `multievolve/featurizers/model_choices.py` 的 `FEATURIZE_CHOICES` 和 `FEATURE_MODELS` 中。具体预训练模型名称来自 `multievolve/featurizers/model_locations.py`。

### 3.2 Predictors：训练和预测突变体性能

`predictors` 负责从特征矩阵学习一个函数：

```text
蛋白序列/突变特征 -> 实验测量值或适应度分数
```

项目内置的预测模型分三类。

传统回归模型在 `base_regressors.py` 中：

| 模型 | 类 | 说明 |
|---|---|---|
| Identity | `IdentityRegressor` | 返回固定值，更多是基线/占位用途。 |
| Linear Regression | `LinearRegressor` | 普通线性回归。 |
| Random Forest | `RandomForestRegressor` | 随机森林回归。 |
| Ridge | `RidgeRegressor` | 带 L2 正则的线性模型，可通过交叉验证选择 alpha。 |

高斯过程模型在 `gaussian_process_regressors.py` 中：

| 模型 | 类 | 说明 |
|---|---|---|
| 普通 GP | `GPRegressor` | 高斯过程回归，可输出不确定性。 |
| 稀疏 GP | `SparseGPRegressor` | 用近似方式降低普通 GP 的计算压力。 |
| 线性核 GP | `GPLinearRegressor` | 使用 linear kernel。 |
| 二次核 GP | `GPQuadRegressor` | 使用 quadratic-style kernel。 |
| RBF 核 GP | `GPRBFRegressor` | 使用 RBF kernel。 |

神经网络模型在 `neural_net_regressors.py` 中：

| 模型 | 类 | 说明 |
|---|---|---|
| 全连接网络 | `Fcn` | 把特征展平后经过多层 Linear、LeakyReLU、Dropout，最后输出一个预测值。 |
| 卷积网络 | `Cnn` | 把序列长度和编码维度看成二维输入，用 `Conv2d` 抽取局部模式。 |

### 3.3 Splitters：决定如何评估模型

`splitters` 负责把数据分成训练集、验证集和测试集。不同切分方式对应不同科学问题。

| 切分方式 | 类 | 适合回答的问题 |
|---|---|---|
| K 折 | `KFoldProteinSplitter` | 模型在小数据集上是否稳定？ |
| 随机切分 | `RandomProteinSplitter` | 普通随机训练/测试表现如何？ |
| 按实验轮次 | `RoundProteinSplitter` | 用早期轮次能否预测后期轮次？ |
| 按突变位点 | `PositionProteinSplitter` | 模型能否外推到没见过的位置？ |
| 按区域 | `RegionProteinSplitter` | 模型能否外推到没见过的蛋白区域？ |
| 按性质值 | `PropertyProteinSplitter` | 模型在高值或低值区域是否可靠？ |
| 按突变负载 | `MutLoadProteinSplitter` | 用少突变数据能否预测多突变体？ |
| 按三维距离 | `ResidueDistanceSplitter` | 模型对结构空间上的外推能力如何？ |

### 3.4 Proposers：生成候选突变体

`proposers` 负责生成候选突变体，并且可以把候选交给模型评分。

| 方法 | 类 | 说明 |
|---|---|---|
| Alanine scanning | `AlanineScanningProposer` | 把每个位点突变为丙氨酸，做快速扫描。 |
| Deep mutational scanning | `DeepMutationalScanningProposer` | 枚举所有单点突变。 |
| Random mutagenesis | `RandomMutagenesisProposer` | 随机生成突变体。 |
| Combinatorial proposer | `CombinatorialProposer` | 从给定突变池枚举组合突变，是论文主流程的关键提议器。 |
| Model-guided proposer | `ModelGuidedProposer` | 以模型预测为指导搜索突变空间。 |
| Simulated annealing | `SimulatedAnnealingProposer` | 用模拟退火寻找高分突变体。 |

## 4. 论文主流程为什么选择这条链

虽然项目内置了很多方法，但命令行主流程实际选择的是一条相对稳健、依赖较少、适合小数据的链：

```text
KFoldProteinSplitter
  -> OneHotFeaturizer
  -> Fcn
  -> CombinatorialProposer
  -> MultiAssemblyDesigner
```

这条链的逻辑是：

首先，训练数据通常不大。对于小数据，`OneHotFeaturizer` 是最直接、最稳定、最少外部依赖的表示方式。它不需要下载 ESM、Ankh、ProtT5 这类大模型，也不需要 MSA 文件或结构文件。

其次，使用 `KFoldProteinSplitter` 是为了减少偶然性。单次随机划分可能刚好让测试集简单或困难，而 K 折会让模型在多个划分上接受检验，更适合比较超参数和模型结构。

然后，`Fcn` 适合处理 one-hot 展平后的特征。它把序列编码矩阵展平成向量，再经过多层全连接网络。项目用 sweep 配置系统搜索层数、隐藏层大小、学习率和 batch size。

训练过程中，WandB 负责记录每个模型结构和超参数在不同 fold 上的表现。这里的 WandB 不是最终候选突变文件的存储位置，而是训练实验记录和超参数选择的中间层。

接着，`p2_propose.py` 会读取 WandB 中同一实验名下的历史 run，找到平均测试表现最好的结构。它不会直接拿 WandB 上的模型权重继续用，而是用选出的最佳超参数在本地重新训练多个模型，形成一个 ensemble。

最后，`CombinatorialProposer` 从突变池枚举组合突变，并让多个模型一起打分。使用 ensemble 平均分比单个模型更稳健，也更符合多突变体设计的目标。`MultiAssemblyDesigner` 再把最终突变列表转换成湿实验可以直接使用的 oligo 表。

与这条监督学习主线并行，`plm_zeroshot_ensemble.py` 是另一条支线。它用 ESM 和 ESM-IF 对所有单点突变直接打分，适合在实验迭代早期或需要补充单点候选时使用。

## 5. 完整数据流：从训练数据到实验设计

### 5.1 第一步：`scripts/p1_train.py` 训练模型

`p1_train.py` 是监督学习主流程的第一步。它的输入来自命令行：

```bash
p1_train.py \
  --experiment-name multievolve_example \
  --protein-name example_protein \
  --wt-files apex.fasta \
  --training-dataset-fname example_dataset.csv \
  --wandb-key <your_wandb_key> \
  --mode test
```

这些参数分别表示：

| 参数 | 数据含义 |
|---|---|
| `experiment-name` | WandB project 名，也是后续 `p2_propose.py` 查找训练 run 的关键名字。 |
| `protein-name` | 蛋白名，用于组织本地缓存目录。 |
| `wt-files` | 野生型氨基酸 FASTA，可以是单链，也可以是逗号分隔的多链。 |
| `training-dataset-fname` | 训练数据 CSV，通常包含突变字符串和实验性质值。 |
| `wandb-key` | 用于登录 WandB。 |
| `mode` | `test` 或 `standard`，决定 sweep 配置规模。 |

脚本内部先调用 `parse_args()` 解析参数，然后执行：

```python
wandb.login(key=args.wandb_key)
```

这一步让后续训练 run 可以写入 WandB。

随后脚本构造数据切分器：

```python
fold_splitter = KFoldProteinSplitter(
    protein_name,
    training_dataset_fname,
    wt_files,
    csv_has_header=True,
    use_cache=True,
    y_scaling=True,
    val_split=0.15,
)
splits = fold_splitter.generate_splits(n_splits=5)
```

这里的数据流是：

```text
training_dataset.csv + wt FASTA
  -> KFoldProteinSplitter
  -> 5 个 split 对象
  -> 每个 split 中包含 X_train, y_train, X_val, y_val, X_test, y_test
```

接着创建特征化器：

```python
onehot = OneHotFeaturizer(protein=protein_name, use_cache=True)
features = [onehot]
```

注意：`OneHotFeaturizer` 此时只是一个对象，还没有真正把所有序列编码。真正编码发生在模型训练时，也就是模型调用 `featurizer.featurize(X)` 的时候。因为 `use_cache=True`，特征结果可以写入本地 `proteins/<protein>/feature_cache/onehot`。

然后脚本指定模型：

```python
models = [Fcn]
```

最后调用：

```python
run_nn_model_experiments(
    splits,
    features,
    models,
    experiment_name=experiment_name,
    use_cache=False,
    sweep_depth=sweep_depth,
    search_method=search_method,
    show_plots=True,
)
```

`run_nn_model_experiments()` 位于 `multievolve/predictors/neural_net_regressors.py`。它会根据模型名、运行模式、搜索方式选择 YAML 文件，例如：

```text
Fcn + test + test -> fcn_test_sweep.yaml
Fcn + standard + grid -> fcn_standard_grid_sweep.yaml
```

之后它创建 WandB sweep：

```python
sweep_id = wandb.sweep(sweep=sweep_config, project=experiment_name)
```

每个 sweep run 都会进入：

```python
with wandb.init() as run:
    config = run.config
    instance = model(split, feature, use_cache=use_cache, config=config, show_plots=show_plots)
    stat = instance.run_model()
```

这里的数据流非常重要：

```text
sweep YAML
  -> WandB 生成一组 run.config
  -> run.config 传给 Fcn
  -> Fcn 根据 config 设置 batch_size、learning_rate、layer_size、num_layers、optimizer、epochs
  -> Fcn.run_model() 训练/验证/测试
  -> 训练 loss、验证 loss、测试指标记录到 WandB
```

`Fcn.run_model()` 内部会使用 `TorchDataProcessor` 把 split 中的数据转成 PyTorch `DataLoader`。训练过程中，模型每个 epoch 会计算训练 loss 和验证 loss，并通过：

```python
wandb.log({"Train Loss": train_loss, "Val Loss": val_loss})
```

记录到 WandB。测试指标和图像则通过 `multievolve/utils/other_utils.py` 中的 `log_results()` 记录。

所以，`p1_train.py` 的输出不是单一文件，而是三类结果：

| 输出位置 | 内容 |
|---|---|
| WandB | 每个 run 的超参数、训练 loss、验证 loss、测试指标、图像等。 |
| 本地 `feature_cache` | 已计算过的 one-hot 或其他特征缓存。 |
| 本地 `model_cache` | 如果启用缓存，保存神经网络权重 `.pth`。 |

### 5.2 第二步：`scripts/p2_propose.py` 选择最佳模型并提议组合突变

`p2_propose.py` 的输入来自命令行：

```bash
p2_propose.py \
  --experiment-name multievolve_example \
  --protein-name example_protein \
  --wt-files apex.fasta \
  --training-dataset example_dataset.csv \
  --mutation-pool combo_muts.csv \
  --top-muts-per-load 3 \
  --export-name multievolve_proposals
```

它读取的本地文件包括：

| 输入 | 用途 |
|---|---|
| `training-dataset` | 重新训练最佳结构的模型。 |
| `wt-files` | 拼接得到野生型氨基酸序列 `wt_seq`。 |
| `mutation-pool` | 候选单点突变池，用于组合成多突变体。 |

脚本先本地读取突变池：

```python
mutation_pool = pd.read_csv(mutation_pool_fname, header=None).values.flatten().tolist()
```

再读取野生型序列：

```python
wt_seq = "".join([str(SeqIO.read(wt_file, "fasta").seq.upper()) for wt_file in wt_files])
```

接下来它不是从本地 CSV 读取训练评估结果，而是从 WandB 读取 `p1_train.py` 产生的历史 run：

```python
api = wandb.Api()
runs = api.runs(experiment_name)
```

对每个 run，它读取两个部分：

```python
run.summary._json_dict
run.config
```

`summary` 中包含测试损失、Pearson、Spearman 等训练结果；`config` 中包含 batch size、learning rate、hidden layer size、num layers、feature 等超参数。脚本会把这些信息整理成一个 DataFrame，然后构造一列 `condition`：

```text
batch_size | learning_rate | layer_size | num_layers | Feature
```

这列表示“同一种模型架构和特征设置”。脚本会跨不同 split/fold 聚合这些条件的表现，按测试损失排名，选出最好的结构。

选出最佳结构后，脚本得到：

```text
bs      -> batch_size
lr      -> learning_rate
hidden  -> layer_size
layers  -> num_layers
```

然后构造新的配置：

```python
config = {
    "layer_size": hidden,
    "num_layers": layers,
    "learning_rate": lr,
    "batch_size": bs,
    "optimizer": "adam",
    "epochs": 300,
}
```

这一步之后，数据流又回到本地。脚本重新构造：

```python
split = KFoldProteinSplitter(...)
splits = split.generate_splits(n_splits=10)
feature = OneHotFeaturizer(protein=protein_name, use_cache=True)
```

然后训练多个模型：

```python
models = []
for split in splits:
    model = Fcn(split, feature, config=config, use_cache=True)
    model.run_model()
    models.append(model)
```

注意这里的模型列表 `models` 是一个 ensemble。它们使用同一组最佳超参数，但在不同 split 上训练。后续候选突变会被多个模型一起打分。

接下来初始化组合突变提议器：

```python
proposer = CombinatorialProposer(
    start_seq=wt_seq,
    models=models,
    trust_radius=11,
    num_seeds=-1,
    mutation_pool=mutation_pool,
)
```

它的输入是：

| 输入 | 含义 |
|---|---|
| `start_seq` | 野生型序列。 |
| `models` | 重新训练得到的 `Fcn` ensemble。 |
| `trust_radius` | 最大组合突变搜索范围。 |
| `num_seeds=-1` | 评估所有 seeds。 |
| `mutation_pool` | 可用于组合的单点突变列表。 |

然后执行：

```python
proposer.propose(output_df=False)
proposer.evaluate_proposals()
proposer.save_proposals(f"{experiment_name}_proposals_all")
```

这里的数据流是：

```text
突变池 + 野生型序列
  -> CombinatorialProposer 枚举组合突变
  -> 每个候选突变体变成序列
  -> models 中每个 Fcn 调用 predict()
  -> 每个候选得到多个预测分数
  -> 取平均分 average
  -> 按 average 排序
```

最终输出包括：

| 输出 | 用途 |
|---|---|
| `<experiment_name>_proposals_all.csv` | 完整候选突变预测结果。 |
| `<experiment_name>_proposals_top_<N>.csv` | 每个突变负载下 top 候选。 |
| `<export_name>.csv` | 只包含突变字符串，供 `p3_assembly_design.py` 读取。 |
| 多链版本 `<export_name>_<chain>_mutants.csv` | 多链蛋白时按链导出。 |

### 5.3 第三步：`scripts/p3_assembly_design.py` 生成寡核苷酸设计

`p3_assembly_design.py` 的输入来自第二步输出：

```bash
p3_assembly_design.py \
  --mutations-file multievolve_proposals.csv \
  --wt-fasta APEX_33overhang.fasta \
  --overhang 33 \
  --species human \
  --oligo-direction bottom \
  --tm 80 \
  --output design
```

这里的关键输入是：

| 输入 | 含义 |
|---|---|
| `mutations-file` | `p2_propose.py` 输出的候选突变 CSV。 |
| `wt-fasta` | 带 overhang 的野生型 DNA 序列。 |
| `overhang` | 两端 overhang 长度。 |
| `species` | 用于选择密码子偏好的物种。 |
| `oligo-direction` | oligo 方向，`top` 或 `bottom`。 |
| `tm` | 设计 oligo 时使用的 melting temperature。 |

脚本本身很薄，真正工作由 `MultiAssemblyDesigner` 完成：

```python
designer = MultiAssemblyDesigner(
    df,
    args.wt_fasta,
    args.overhang,
    args.species,
    oligo_direction=args.oligo_direction,
    tm=args.tm,
    output=args.output,
)
```

`MultiAssemblyDesigner` 位于 `multievolve/utils/cloning_utils.py`。它的核心流程是：

```text
候选氨基酸突变
  -> 找到对应密码子位置
  -> 根据物种密码子偏好选择突变密码子
  -> 生成突变 DNA 片段
  -> 按 Tm 和方向设计 oligo
  -> 导出实验表
```

最终输出通常在 `wt-fasta` 所在目录下：

| 输出 | 内容 |
|---|---|
| `cloning_sheet.csv` | 每个候选突变体需要哪些 oligo。 |
| `oligos.csv` | 具体 oligo 序列和编号。 |

### 5.4 支线：`scripts/plm_zeroshot_ensemble.py` 用 PLM 提名单点突变

这条流程和前面训练模型的流程相对独立。它不依赖 `p1_train.py` 产生的 WandB run，也不读取 `p2_propose.py` 的 ensemble 模型。

运行示例：

```bash
plm_zeroshot_ensemble.py \
  --wt-file apex.fasta \
  --pdb-files apex.cif \
  --variants 24 \
  --excluded-positions 1,14,41,112 \
  --normalizing-method aa_substitution_type
```

输入包括：

| 输入 | 含义 |
|---|---|
| `wt-file` | 野生型氨基酸 FASTA。 |
| `pdb-files` | PDB/CIF 结构文件，可有多个。 |
| `variants` | 每种方法提名多少个突变。 |
| `excluded-positions` | 不允许突变的位置。 |
| `normalizing-method` | z-score 标准化分组方式。 |

脚本会调用：

```python
zero_shot_esm_dms(wt_seq)
zero_shot_esm_if_dms(wt_seq, pdb_file, chain_id="A", scoring_strategy="wt-marginals")
```

它会产生四类候选：

| 候选来源 | 含义 |
|---|---|
| ESM | 按 ESM raw score 选择。 |
| ESM-IF | 按 ESM-IF raw score 选择。 |
| ESM-z | 对 ESM 分数标准化后选择。 |
| ESM-IF-z | 对 ESM-IF 分数标准化后选择。 |

最终输出：

```text
plm_zeroshot_ensemble_nominated_mutations.csv
```

这个 CSV 包含提名突变以及它们被哪些方法选中。

## 6. Featurizers 细节示例：ESM 如何把序列变成 embedding

ESM 相关代码位于：

```text
multievolve/featurizers/esm_featurizers.py
```

### 6.1 类之间的继承关系

ESM 类的核心结构是：

```text
BaseFeaturizer
  -> ESMBaseFeaturizer
       -> ESMLogitsFeaturizer
       -> ESM1vEmbedFeaturizer
       -> ESM2EmbedFeaturizer
       -> ESM2_15b_EmbedFeaturizer
```

`BaseFeaturizer` 管通用流程，例如缓存、去重、调用 `custom_featurizer()`。`ESMBaseFeaturizer` 管 ESM 模型加载和运行。具体子类只决定默认用哪些 ESM 模型、输出 logits 还是 embedding。

### 6.2 `BaseFeaturizer.featurize()` 的通用流程

所有 featurizer 都遵循这个模板：

```text
输入 seqs
  -> 如果 use_cache=True，先 load_features(seqs)
  -> 找出没缓存过的 unique sequences
  -> 调用子类 custom_featurizer(unique_seqs)
  -> 把新结果写入 seq_to_feature
  -> 如果 use_cache=True，update_cache(...)
  -> 按原始输入顺序组装 np.ndarray
  -> 如果 flatten_features=True，reshape 成二维
  -> 返回 X
```

因此，模型调用的通常只是：

```python
X = featurizer.featurize(sequences)
```

但背后可能包含缓存读取、大模型推理、特征拼接等步骤。

### 6.3 ESM 如何导入和加载

`ESMBaseFeaturizer.featurize_esm()` 内部导入 ESM：

```python
from esm import pretrained
```

然后对每个模型名执行：

```python
model, alphabet = pretrained.load_model_and_alphabet(model_location)
model.eval()
```

`model_location` 来自 `FEATURE_MODELS`。例如：

```python
FEATURE_MODELS["esm_embed_2_3b"]
```

对应的是 `model_locations.py` 中登记的某个 ESM-2 3B 模型名称。

### 6.4 ESM 如何运行

ESM 不能直接吃字符串列表，需要先把序列转成 token。代码使用：

```python
batch_converter = alphabet.get_batch_converter()
batch_labels, batch_strs, batch_tokens = batch_converter(sequence_data)
```

然后调用 `eval_esm()`。根据 `output_type` 不同，输出有两种主要形式：

| `output_type` | 输出含义 |
|---|---|
| `log_probabilities` | 每个位点对氨基酸的 log probability/logits 类特征。 |
| `sequence_representations` | 对 token representation 做平均，得到整条序列 embedding。 |

如果一个 featurizer 配置了多个模型，代码会对多个模型的结果求平均：

```python
np.mean(model_features, axis=0)
```

如果序列太长，代码会分窗口处理，最后用：

```python
X = np.hstack(output)
```

把窗口结果横向拼接。

最终输出是一个 `np.ndarray`。这个数组可以继续进入：

```text
BaseRegressor.preprocess_data()
BaseNN / TorchDataProcessor
CombinatorialFeaturizer
```

### 6.5 ESM 子类的差别

`ESMLogitsFeaturizer` 调用：

```python
self.featurize_esm(seqs, output_type="log_probabilities")
```

输出位置级别的 ESM 分数。

`ESM1vEmbedFeaturizer`、`ESM2EmbedFeaturizer`、`ESM2_15b_EmbedFeaturizer` 调用：

```python
self.featurize_esm(seqs, output_type="sequence_representations")
```

输出序列级 embedding。

这就是“用 ESM 将氨基酸序列编码为 embedding”的代码链条：

```text
ESM2EmbedFeaturizer.custom_featurizer()
  -> ESMBaseFeaturizer.featurize_esm()
  -> from esm import pretrained
  -> pretrained.load_model_and_alphabet(...)
  -> alphabet.get_batch_converter()
  -> eval_esm()
  -> np.ndarray embedding
```

## 7. Base regressors 细节示例：传统模型如何读取数据、设置参数和输出结果

传统回归模型位于：

```text
multievolve/predictors/base_regressors.py
```

### 7.1 `BaseRegressor` 读入什么数据

`BaseRegressor` 初始化时接收两个关键对象：

```python
BaseRegressor(data_splitter, featurizer, ...)
```

`data_splitter` 提供数据：

```text
X_train, y_train
X_test, y_test
```

如果是神经网络，还会用到验证集；传统 `BaseRegressor` 主要使用训练集和测试集。

`featurizer` 提供编码方法：

```text
序列字符串 -> 数值矩阵
```

所以 `BaseRegressor` 本身并不关心序列如何编码。它只知道调用：

```python
self.featurizer.featurize(X)
```

### 7.2 `BaseRegressor` 的模板方法结构

`BaseRegressor` 是典型的模板方法结构。父类定义完整流程，子类只补具体模型。

父类负责：

```text
run_model()
load_model()
save_model()
featurize()
preprocess_data()
evaluate()
predict()
```

子类必须实现：

```text
train()
custom_predictor()
```

也就是说，所有传统模型都共享同一个数据预处理、缓存、评估和预测接口，但训练细节由子类决定。

### 7.3 `run_model()` 的数据流

`run_model()` 的逻辑是：

```text
如果 use_cache=True 且模型文件存在
  -> 加载缓存模型
否则
  -> preprocess_data(X_train)
  -> train(X, y_train)
  -> 如开启缓存则 save_model()

如果 eval=True
  -> evaluate()
  -> 返回测试指标 dict
```

模型缓存路径通常位于：

```text
proteins/<protein>/model_cache/<dataset>/objects/<model_name>.pkl
```

### 7.4 `preprocess_data()` 做什么

`preprocess_data()` 会先调用 featurizer：

```python
X = self.featurizer.featurize(X)
```

然后把输出 reshape 成二维：

```python
X = X.reshape(X.shape[0], -1)
```

这是因为 scikit-learn 的传统模型通常需要二维输入：

```text
样本数 x 特征数
```

如果 one-hot 原本是：

```text
样本数 x 序列长度 x 氨基酸类别数
```

这里就会被展平成：

```text
样本数 x (序列长度 * 氨基酸类别数)
```

### 7.5 子类如何设置模型

`LinearRegressor` 使用 scikit-learn 的线性回归：

```python
model = LinearRegression(...)
self.model = model
self.model.fit(X, y)
```

预测时：

```python
return self.model.predict(X)
```

`RandomForestRegressor` 使用 sklearn 随机森林。它会把初始化参数传给：

```python
RFRegressor(
    n_estimators=self.n_estimators,
    criterion=self.criterion,
    max_depth=self.max_depth,
    min_samples_split=self.min_samples_split,
    min_samples_leaf=self.min_samples_leaf,
    max_features=self.max_features,
    ...
)
```

再执行：

```python
self.model.fit(X, y)
```

`RidgeRegressor` 如果用户没有指定 `reg_coef`，会遍历 `reg_coef_list`，用 `cross_val_score()` 做交叉验证，选择表现最好的 alpha。之后创建：

```python
self.linear_model_cls(alpha=best_reg_coef)
```

默认的 `linear_model_cls` 是 sklearn 的 `Ridge`。

### 7.6 `evaluate()` 输出什么

`evaluate()` 会对测试集调用：

```python
y_pred = self.predict(self.X_test)
```

然后调用：

```python
performance_report(y, y_pred)
```

输出 Pearson、Spearman 等指标。它还会画一个“预测值 vs 真实值”的散点图，并把指标写在图上。最后返回一个统计指标字典。

如果在批量实验中使用 `run_model_experiments()`，多个模型、多个特征、多个 split 的结果会被整理成一个 DataFrame；如果 `use_cache=True`，还会保存到：

```text
model_cache/<dataset>/results/<experiment_name>.csv
```

## 8. 神经网络模型：`BaseNN`、`Fcn` 和 `Cnn`

论文主流程使用的是 `Fcn`，它位于：

```text
multievolve/predictors/neural_net_regressors.py
```

### 8.1 `run_nn_model_experiments()` 是如何被调用的

`p1_train.py` 中：

```python
models = [Fcn]
run_nn_model_experiments(splits, features, models, experiment_name=...)
```

`run_nn_model_experiments()` 遍历：

```text
每个 split
  -> 每个 feature
  -> 每个 model class
```

然后选择对应的 sweep YAML，创建 WandB sweep，并让 WandB agent 逐个运行超参数组合。

### 8.2 `BaseNN` 做什么

`BaseNN` 继承自 `torch.nn.Module`。它负责通用训练逻辑：

```text
接收 data_splitter 和 featurizer
  -> 用 TorchDataProcessor 构造 DataLoader
  -> 设置 optimizer 和 loss
  -> train_loop()
  -> val_loop()
  -> early stopping
  -> test_loop()
  -> 记录结果
  -> 可保存 .pth 权重
```

`BaseNN` 的模型缓存路径通常是：

```text
proteins/<protein>/model_cache/<dataset>/objects/<model_name>.pth
```

### 8.3 `Fcn` 的结构

`Fcn` 的构造函数从 `config` 中读取：

```text
layer_size
num_layers
learning_rate
batch_size
optimizer
epochs
```

它先用一个训练样本经过 featurizer，判断输入维度：

```python
X_train_feat_example = self.nn_data_processor.featurize([self.nn_data_processor.X_train[0]])[0]
input_features = X_train_feat_example.flatten().shape[0]
```

然后构建网络：

```text
Flatten
  -> Linear(input_features, layer_size)
  -> LeakyReLU
  -> Dropout
  -> 若干 Linear(layer_size, layer_size) + LeakyReLU + Dropout
  -> Linear(layer_size, 1)
```

所以 `Fcn` 最终输出一个数字，表示模型预测的突变体性能。

### 8.4 `Cnn` 的结构

`Cnn` 把特征矩阵看成二维输入：

```text
序列长度 x 编码维度
```

它先 `unsqueeze(1)` 增加 channel 维度，然后经过多层 `nn.Conv2d`，再 flatten，最后用全连接层输出一个预测值：

```text
Conv2d 层
  -> LeakyReLU
  -> Flatten
  -> Linear
  -> LeakyReLU
  -> Dropout
  -> Linear 输出 1 个值
```

主流程默认没有用 `Cnn`，但项目已经提供了对应 sweep 配置。

## 9. 函数和类之间如何引用

这个项目大量使用星号导入，例如：

```python
from multievolve.splitters import *
from multievolve.featurizers import *
from multievolve.predictors import *
from multievolve.proposers import *
```

这会让新手有点难追踪类来自哪里。实际路径如下：

| 脚本中看到的名字 | 实际来源 |
|---|---|
| `KFoldProteinSplitter` | `multievolve/splitters/base_splitters.py` |
| `OneHotFeaturizer` | `multievolve/featurizers/base_featurizers.py` |
| `ESM2EmbedFeaturizer` | `multievolve/featurizers/esm_featurizers.py` |
| `Fcn` | `multievolve/predictors/neural_net_regressors.py` |
| `run_nn_model_experiments` | `multievolve/predictors/neural_net_regressors.py` |
| `LinearRegressor`、`RandomForestRegressor`、`RidgeRegressor` | `multievolve/predictors/base_regressors.py` |
| `CombinatorialProposer` | `multievolve/proposers/base_proposers.py` |
| `MultiAssemblyDesigner` | `multievolve/utils/cloning_utils.py` |
| `zero_shot_esm_dms`、`zero_shot_esm_if_dms` | `multievolve/utils/zeroshot_utils.py` |

这些名字能被直接导入，是因为每个子包的 `__init__.py` 会再次导出具体文件中的类。顶层 `multievolve/__init__.py` 又导出了所有子包：

```python
from multievolve.splitters import *
from multievolve.predictors import *
from multievolve.proposers import *
from multievolve.utils import *
from multievolve.featurizers import *
```

因此，`scripts/p3_assembly_design.py` 中可以写：

```python
from multievolve import MultiAssemblyDesigner
```

虽然 `MultiAssemblyDesigner` 实际定义在：

```text
multievolve/utils/cloning_utils.py
```

## 10. 本地文件、缓存和远程记录如何一起工作

整个项目的数据流不是只靠一个目录，也不是只靠 WandB。它同时使用本地文件、本地缓存和 WandB 记录。

### 10.1 原始输入文件

| 文件类型 | 例子 | 用途 |
|---|---|---|
| 氨基酸 FASTA | `apex.fasta` | 提供野生型蛋白序列。 |
| DNA FASTA | `APEX_33overhang.fasta` | 提供寡核苷酸设计所需 DNA 序列。 |
| 训练 CSV | `example_dataset.csv` | 提供突变字符串和实验性质值。 |
| 突变池 CSV | `combo_muts.csv` | 提供可组合的单点突变。 |
| 结构文件 | `apex.cif` | 供 ESM-IF 或结构距离相关方法使用。 |

### 10.2 本地缓存

项目会自动在类似下面的路径下生成缓存：

```text
proteins/<protein_name>/
  feature_cache/
  model_cache/
  split_cache/
  proposers/results/
```

这些缓存的作用是：

| 缓存 | 内容 |
|---|---|
| `feature_cache` | 序列到特征矩阵的映射，例如 one-hot 或 ESM embedding。 |
| `split_cache` | 数据切分结果，避免重复生成 split。 |
| `model_cache` | 训练好的模型，传统模型是 `.pkl`，神经网络是 `.pth`。 |
| `proposers/results` | 候选突变体和模型打分结果。 |

### 10.3 WandB 记录

WandB 出现在训练和提议突变之间的数据流中：

```text
p1_train.py
  -> run_nn_model_experiments()
  -> wandb.sweep()
  -> wandb.init()
  -> 记录每个 run 的 config 和 summary

p2_propose.py
  -> wandb.Api()
  -> api.runs(experiment_name)
  -> 读取 run.config 和 run.summary
  -> 选择最佳模型结构
```

换句话说，WandB 在这里保存的是“训练实验记录”：每组超参数、每个 fold、每个模型结构对应的 loss 和测试指标。`p2_propose.py` 需要这些记录来决定哪组超参数最好。

候选突变 CSV、寡核苷酸设计 CSV 等最终文件仍然是本地导出的。

## 11. 一条完整链条的文字版流程图

```text
训练 CSV + 野生型 FASTA
  -> KFoldProteinSplitter 生成多个 split
  -> OneHotFeaturizer 把序列转成 one-hot
  -> run_nn_model_experiments 读取 sweep YAML
  -> WandB sweep 分配超参数
  -> Fcn 在每个 fold/超参数组合上训练
  -> WandB 保存 run.config 和 run.summary

同一个 experiment_name
  -> p2_propose.py 通过 WandB API 读取历史 run
  -> 聚合相同架构在不同 fold 上的测试表现
  -> 选择最佳 batch_size / learning_rate / layer_size / num_layers
  -> 用最佳配置重新训练多个 Fcn
  -> 得到模型 ensemble

野生型序列 + 突变池 + 模型 ensemble
  -> CombinatorialProposer 枚举组合突变
  -> 每个模型预测候选分数
  -> ensemble 平均分排序
  -> 导出候选突变 CSV

候选突变 CSV + 野生型 DNA FASTA
  -> MultiAssemblyDesigner
  -> 密码子选择和 oligo 设计
  -> cloning_sheet.csv + oligos.csv
```

## 12. 新手读代码的建议路线

如果你只是想理解论文主流程，建议按这个顺序读：

1. `scripts/p1_train.py`：理解训练入口。
2. `multievolve/splitters/base_splitters.py`：理解数据如何被切分。
3. `multievolve/featurizers/base_featurizers.py`：理解 one-hot 和 featurizer 通用模板。
4. `multievolve/predictors/neural_net_regressors.py`：理解 `run_nn_model_experiments`、`BaseNN`、`Fcn`。
5. `scripts/p2_propose.py`：理解如何读取 WandB 结果、选择最佳超参数、重新训练 ensemble。
6. `multievolve/proposers/base_proposers.py`：理解 `CombinatorialProposer` 如何生成和评分组合突变。
7. `scripts/p3_assembly_design.py` 和 `multievolve/utils/cloning_utils.py`：理解候选突变如何变成 oligo。

如果你想理解蛋白语言模型支线，再读：

1. `scripts/plm_zeroshot_ensemble.py`
2. `multievolve/utils/zeroshot_utils.py`
3. `multievolve/featurizers/esm_featurizers.py`
4. `multievolve/featurizers/zeroshot_featurizers.py`

如果你想理解项目提供了哪些可替换方法，再读：

1. `multievolve/featurizers/model_choices.py`
2. `multievolve/predictors/base_regressors.py`
3. `multievolve/predictors/gaussian_process_regressors.py`
4. `multievolve/splitters/base_splitters.py`
5. `multievolve/proposers/base_proposers.py`

## 13. 最简记忆版

这个项目可以用下面这条链记住：

```text
数据切分 splitters
  -> 序列编码 featurizers
  -> 模型训练 predictors
  -> 候选突变 proposers
  -> 实验设计 utils.cloning_utils
```

论文命令行主流程实际用的是：

```text
KFoldProteinSplitter
  -> OneHotFeaturizer
  -> Fcn with WandB sweep
  -> WandB 记录并供 p2_propose.py 选择最佳超参数
  -> Fcn ensemble
  -> CombinatorialProposer
  -> MultiAssemblyDesigner
```

zero-shot 支线用的是：

```text
野生型序列 + 结构文件
  -> ESM / ESM-IF
  -> 单点突变打分
  -> 提名单点突变 CSV
```

这就是 MULTI-evolve 代码的基本骨架：它不是一个单独模型，而是一套把实验数据、机器学习预测、蛋白语言模型提名和湿实验构建连接起来的工程化流程。
