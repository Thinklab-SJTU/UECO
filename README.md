# UECO: A Unified Encoder with Structure-Aware Attention Mixture via Iterative Edge Evolving for Neural Combinatorial Optimization**

## Abstract

Despite the promise of Neural Combinatorial Optimization (NCO), recent literature has increasingly coupled decision paradigms, training objectives, and problem classes with specific backbone architectures, limiting generic model design and fair cross-paradigm evaluation. We propose UECO, a simple yet effective Unified Encoder for Combinatorial Optimization with a structure-aware attention mixture scheme, where a lightweight module defined upon graph-based locality is interleaved with conventional global node-focused attention blocks to dynamically evolve edge representations and inject local topological information into attention scores. Instead of fusing static edge features in a single step, UECO integrates local and global messages in a progressive and problem-agnostic manner. UECO works with Local Construction (LC), Global Prediction (GP), and Adaptive Expansion (AE) paradigms, across node- or edge-oriented problems, symmetric or asymmetric tasks, sparse or dense graphs, and supervised, reinforcement, or generative learning. Experiments show that replacing existing encoders with UECO improves solution quality and generalization across TSP, ATSP, CVRP, ACVRP, MIS, MaxClique, and MaxCut at multiple scales.

## Method

UECO jointly maintains node and edge representations. Each layer combines global attention-based node updates with a local structure-aware update that evolves edge states. The updated edge representations are injected into the attention scores at every layer, enabling progressive interaction between global node semantics and local graph topology. Dual-channel representations support directional and asymmetric instances. The resulting encoder produces node or edge embeddings for heatmap prediction, autoregressive construction, and iterative generation.

![UECO in the NCO landscape](figures/fig1.png)

![UECO architecture](figures/fig2.png)

## Installation

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

## Data and weights

- Pretrained weights: [download]()
- Test data: [download]()
- Training data: [ML4CO-Bench-101-SL](https://huggingface.co/datasets/ML4CO/ML4CO-Bench-101-SL/tree/main/train_dataset)

The repository layout is:

```text
weights/{AE,GP,LC}/<task>/
data/test/<TASK>/
data/train/<TASK>/<dataset-folder>/*.txt
```

The included training folders contain one compatible `.txt` shard for a direct smoke test. Download the complete training data from ML4CO-Bench-101-SL for full training runs.

## Evaluation

Run each entry from its task directory. Paths to test data and weights are written directly in each script as paths relative to that directory.

```bash
cd LC/atsp
python test_atsp.py
```

```bash
cd AE/tsp
python test_tsp.py

cd GP/mis
python test_mis.py

cd LC/cvrp
python test_cvrp.py
```

Change `TEST_DATA`, `NODES_NUM`, or the corresponding configuration block in a task script to select another released setting.

## Training

Training is task-local and uses the same manual entry-point style:

```bash
cd GP/tsp
python train_tsp.py
```

Edit the configuration block in the script to select the dataset, GPU, batch size, and number of epochs.
