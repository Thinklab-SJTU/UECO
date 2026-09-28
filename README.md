# UECO

Official implementation of **UECO**, organized around the three solving paradigms used in the paper:

- `AE`: adaptive expansion
- `GP`: global prediction
- `LC`: local construction

UECO replaces the encoder/backbone inside each host pipeline while retaining the pipeline's task-specific environment, decoder, solver, state transition, action space, and feasibility handling. The repository contains only UECO training/testing entries and the released UECO weights. Baseline entries, historical experiment logs, plots, Weights & Biases records, paper/rebuttal files, and an embedded copy of ML4CO-Kit are not included.

## Environment setup

The release was validated with Python 3.8 and the local `bench4co-v2` environment. To create an equivalent Conda environment named `ueco`:

```bash
conda create -n ueco python=3.8 -y
conda activate ueco

# Install the PyTorch/CUDA build appropriate for your machine. The validated
# environment uses PyTorch 2.0.1 with CUDA 11.7.
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu117
pip install torch-scatter==2.1.2 torch-sparse==0.6.18 \
  -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install numpy==1.24.4 scipy==1.10.1 matplotlib==3.7.5 \
  tqdm==4.67.1 tabulate==0.8.10 pytorch-lightning==2.0.1 \
  ml4co-kit==0.3.3 wandb==0.20.1
```

If your CUDA version differs, install matching PyTorch, `torch-scatter`, and `torch-sparse` wheels instead.

## Weights and data

Released checkpoints are collected in the repository-level `weights/` directory, and the test sets used by the manual evaluation entries are collected in `data/`. Both directories are intentionally ignored by Git because they contain large binary/data artifacts. Test entries resolve `../../weights` and `../../data` from their task directories, so no environment variable is required for the included tests.

To override the bundled test-data location or configure external training data, set:

```bash
export UECO_DATA_ROOT=/absolute/path/to/test_and_validation_data
export UECO_TRAIN_DATA_ROOT=/absolute/path/to/training_data
```

The expected subdirectories under `UECO_DATA_ROOT` are `TSP`, `ATSP`, `CVRP`, `CVRPLIB`, `MIS`, `MCl`, and `MCut`; filenames are listed directly in each task entry. AE/GP supervised training folders are resolved relative to `UECO_TRAIN_DATA_ROOT` using the same task subdirectories. To override the bundled CVRPLIB directory, set:

```bash
export UECO_CVRPLIB_ROOT=/absolute/path/to/CVRPLIB/instances
```

## Manual evaluation

Run each task from its own directory. Released weights are stored in the repository-level `weights/` directory. Select the problem size or dataset by editing the configuration block near the top of the entry, as in the original codebase.

```bash
cd LC/atsp
python test_atsp_uni.py
```

Other examples:

```bash
cd AE/tsp
python test_tsp_uni.py

cd GP/mis
python test_mis_uni.py

cd LC/cvrp
python test_cvrp_uni.py
```

The historical `_uni.py` suffix is retained only on manual entry filenames for command compatibility; implementation modules, classes, and released weight names use `UECO`.

TSP and ATSP retain the original ML4CO-Kit 2-opt paths used by their host pipelines. The later custom/compiled ATSP "new 2-opt" extension is intentionally not included.

## Manual training

Training follows the same per-task workflow and intentionally has no integrated reproduction launcher:

```bash
cd GP/tsp
python train_tsp_uni.py
```

Edit the task configuration block to choose the dataset, GPU devices, batch size, and epoch count. AE/GP retain the original ML4CO-Kit Weights & Biases logger and write new checkpoints beneath `train_ckpts/`; authenticate with `wandb login` before training. Historical WandB runs and log directories are not included. LC retains the original task-local logger/checkpoint behavior of its host construction pipeline.
