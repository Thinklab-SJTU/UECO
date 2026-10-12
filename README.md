# UECO: A Unified Encoder for Neural Combinatorial Optimization with Iterative Edge-Node Reasoning

This is the official implementation of our NeurIPS 2026 paper "[UECO: A Unified Encoder for Neural Combinatorial Optimization with Iterative Edge-Node Reasoning](https://openreview.net/forum?id=M2lqt23VO7)" | [![NeurIPS 2026 Oral](https://img.shields.io/badge/NeurIPS-2026%20Oral-b31b1b.svg)](paper_preview.pdf) [[paper preview*](paper_preview.pdf)]

**Authors**: [**Wenzheng Pan**](https://wzever.github.io/) · [**Shuyi Yan**](https://www.cozease.cn/) · [**Nuoyan Chen**](https://cny123222.github.io/about/) · [**Yiyang Qu**](https://github.com/Qyy12345) · [**Jiale Ma**](https://heatingma.github.io/) · [**Junchi Yan**](https://thinklab.sjtu.edu.cn)

*The current PDF is provided for early readers only; please refer to the officially published camera-ready version for the definitive paper.

## Abstract

Despite the promise of Neural Combinatorial Optimization (NCO), recent literature has increasingly coupled decision paradigms, training objectives, and problem classes with specific backbone architectures, limiting generic model design and fair cross-paradigm evaluation. We propose UECO, a simple yet effective Unified Encoder for Combinatorial Optimization with a structure-aware attention mixture scheme, where a lightweight module defined upon graph-based locality is interleaved with conventional global node-focused attention blocks to dynamically evolve edge representations and inject local topological information into attention scores. Instead of fusing static edge features in a single step, UECO integrates local and global messages in a progressive and problem-agnostic manner. UECO works with Local Construction (LC), Global Prediction (GP), and Adaptive Expansion (AE) paradigms, across node- or edge-oriented problems, symmetric or asymmetric tasks, sparse or dense graphs, and supervised, reinforcement, or generative learning. Experiments show that replacing existing encoders with UECO improves solution quality and generalization across TSP, ATSP, CVRP, ACVRP, MIS, MaxClique, and MaxCut at multiple scales.

## 🌐 Why a Unified Encoder?

Neural combinatorial optimization has developed along several successful but largely separated streams. Local Construction (LC), Global Prediction (GP), and Adaptive Expansion (AE) methods are commonly paired with different learning objectives, task families, and neural backbones. This empirical coupling makes architectural choices a confounding factor: moving to a new solving paradigm often requires redesigning the representation model as well, obscuring controlled comparison and limiting transfer across problems.

UECO decouples **representation learning** from **decision making**. It provides a **shared, plug-and-play encoder** that preserves the long-range relational modeling of attention while explicitly capturing the local topology carried by graph edges. The same backbone can therefore serve **autoregressive policies**, **one-shot heatmap predictors**, and **iterative generative solvers** across *node- and edge-oriented*, *dense and sparse*, *symmetric and asymmetric* CO problems.

![UECO in the NCO landscape](figures/fig1.png)

## 🧩 Iterative Edge-Node Reasoning

UECO jointly maintains node and edge representations through two complementary components. A **global node-centric attention module** captures long-range dependencies, while a lightweight **local structure-aware module** aggregates neighborhood information and evolves edge states. Rather than injecting fixed edge features once, UECO interleaves these components layer by layer: **locally refined edge embeddings dynamically modulate the next global attention scores, and the updated node semantics in turn guide subsequent edge evolution**.

Modality-specific embedders support node features, edge features, or both. For directional and asymmetric instances, dual node channels retain the two endpoint views of each edge before their structural messages are merged. The final representations remain decoder-agnostic and can be consumed by global heatmap prediction, autoregressive next-action selection, or iterative solution generation.

<p align="center">
  <img src="figures/fig2.png" width="68%" alt="UECO architecture">
</p>


## ⚙️ Installation

```bash
conda create --name ml4co python=3.8
conda activate ml4co

pip install torch==2.1.0
pip install scipy==1.10.1
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-2.1.0+cu121.html

pip install ml4co-kit==0.3.3
pip install wandb==0.16.3
pip install tensordict==0.2.0
pip install pytorch-lightning==2.1.0
pip install einops==0.8.0
pip install matplotlib tqdm scikit-learn
```

## 📦 Data and weights

- Pretrained weights: [download](https://drive.google.com/file/d/1j3iD6Yv0pG97RsOgnU-jdiFhPKaNipUW/view?usp=sharing)
- Test data: [download](https://drive.google.com/file/d/1tLQBUBAMJLIcW3KtS5gSXzuuKmdG9VOD/view?usp=share_link)
- Training data: [ML4CO-Bench-101-SL](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/tree/main/train_dataset)

The repository layout is:

```text
{AE,GP,LC}/train_scripts/train_ueco_<paradigm>_<task>.py
{AE,GP,LC}/test_scripts/test_ueco_<paradigm>_<task>.py
weights/{AE,GP,LC}/<task>/*.pt
data/test/<TASK>/*.txt
data/train/<TASK>/<dataset-folder>/*.txt
```

Download the complete data and unzip them for full runs.

## 🧪 Evaluation

Enter a paradigm directory and run the desired task entry directly. Paths to test data and weights are written explicitly in each script.

### Adaptive Expansion (AE)

```bash
cd AE

python test_scripts/test_ueco_ae_tsp.py
python test_scripts/test_ueco_ae_mis.py
python test_scripts/test_ueco_ae_mcl.py
python test_scripts/test_ueco_ae_mcut.py
```

The default AE evaluation uses `INFERENCE_STEP = 1`; only `DETERMINATE_STEP` is varied across task settings. To evaluate the generative `GP-SL (Gen)` setting reported in the paper, set `DETERMINATE_STEP = 1` and increase `INFERENCE_STEP` (for example, `INFERENCE_STEP = 5`) in the selected AE test script.

### Global Prediction (GP)

```bash
cd GP

python test_scripts/test_ueco_gp_tsp.py
python test_scripts/test_ueco_gp_mis.py
python test_scripts/test_ueco_gp_mcl.py
python test_scripts/test_ueco_gp_mcut.py
```

### Local Construction (LC)

```bash
cd LC

python test_scripts/test_ueco_lc_tsp.py
python test_scripts/test_ueco_lc_atsp.py
python test_scripts/test_ueco_lc_cvrp.py
```

Change `TEST_DATA`, `NODES_NUM`, or the corresponding configuration block in a task script to select another released setting.

## 🏋️ Training

Training uses the same paradigm-local entry-point style. As above, each command is independent.

### Adaptive Expansion (AE)

```bash
cd AE

python train_scripts/train_ueco_ae_tsp.py
python train_scripts/train_ueco_ae_mis.py
python train_scripts/train_ueco_ae_mcl.py
python train_scripts/train_ueco_ae_mcut.py
```

### Global Prediction (GP)

```bash
cd GP

python train_scripts/train_ueco_gp_tsp.py
python train_scripts/train_ueco_gp_mis.py
python train_scripts/train_ueco_gp_mcl.py
python train_scripts/train_ueco_gp_mcut.py
```

### Local Construction (LC)

```bash
cd LC

python train_scripts/train_ueco_lc_tsp.py
python train_scripts/train_ueco_lc_atsp.py
python train_scripts/train_ueco_lc_cvrp.py
```

Edit the configuration block in the script to select the dataset, GPU, batch size, and number of epochs.


## 📚 Acknowledgements

The implementations partly build on [MatNet](https://github.com/yd-kwon/MatNet), [ML4CO-Kit](https://github.com/Thinklab-SJTU/ML4CO-Kit), and [ML4CO-Bench-101](https://github.com/Thinklab-SJTU/ML4CO-Bench-101).