
# RheoSparse: Exploring Finer-Grained Structured Sparsity for Small Language Models

This repo contains code and artifacts for **RheoSparse** (camera-ready release):

- **Sparsity search / pruning / assembly** scripts for HF-style models.
- A CUDA **Sparse Linear** kernel and an **end-to-end inference demo**.
- Evaluation scripts (perplexity and sparsity checks).

## 📁 Repository layout (quick view)

### Model artifacts

- HF-style pruned/searched/assembled model weights (not compressed):
	- `EXPERIMENTS/llama-1b/`
	- `EXPERIMENTS/opt-1.3b/`
	- `EXPERIMENTS/qwen2.5/`

- Compressed sparse weights used by the inference demo:
	- `sparse_infer/sparse_weights_32_passed.npy`

### Inference kernel and demos

- CUDA extension: `sparse_infer/Sparse_Linear/`
- Demo entrypoints: `sparse_infer/qwen_demo.py`, `sparse_infer/gemma_demo.py`

## ⚙️ Environment

Dependencies are listed in `requirements.txt`.

Notes:
- The CUDA extension is built via `pip install -e .` in `sparse_infer/Sparse_Linear/`.
- `sparse_infer/Sparse_Linear/command.sh` exports CUDA/GCC toolchain environment variables.
	Adjust CUDA path, GCC version, and GPU arch if your setup differs.

## 🚀 How to run

### 1) End-to-end inference demo

Build and install the CUDA extension (Sparse Linear kernel):

```bash
cd sparse_infer/Sparse_Linear
bash command.sh
```

Run the demo (Qwen example):

```bash
cd ..
bash run_demo.sh
```

`run_demo.sh` currently runs:

```bash
python qwen_demo.py \
	--model ../EXPERIMENTS/qwen2.5/qwen2.5/ \
	--sparse_model sparse_weights_32_passed.npy \
	--sparse_path ../EXPERIMENTS/qwen2.5/qwen2.5-3264-searched/32-gene_info.jsonl \
	--base_sparsity 32 \
	--num_layers 28
```

If you use a different GPU, you may need to edit `sparse_infer/Sparse_Linear/setup.py` and change:

- `-arch=sm_75` to your GPU's SM version (e.g., `sm_80`, `sm_86`, `sm_90`).

### 2) Evaluation

- `ppl_eval.py`: evaluate perplexity (WikiText-2).
- `checkSparsity.py`: validate overall sparsity.

## 🧰 Other scripts

- `generate_multiple_sparsities.py`: calls `prune.py` to generate multiple fine-grained sparsified models (for search).
- `batch_search.py`: searches the optimal sparsity combination under an overall sparsity constraint.
- `batched_model_assemble.sh`: assembles a HF-style model for evaluation.
- `batched_model_evaluation.sh`: evaluates model performance via `lm_eval`.



