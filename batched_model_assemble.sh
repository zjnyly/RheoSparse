# #!/bin/bash

# # 设置 GPU ID，例如：7
# GPU_ID=0

# # base_sparsity 范围：从 20 到 44（含）
# START_SPARSITY=32
# END_SPARSITY=32

# # 固定参数
# SPARSE_MODEL_PATH="/home/dac/zjn_data/llama-1b-pruned/"
# FULL_MODEL_PATH="/home/dac/zjn_data/llama-1b/"
# LAYER_NUM=16
# # GENE_FOLDER="llama-1b-searched-3-stage-all/"
# GENE_FOLDER="llama_1b_attn_ffn/"
# OUTPUT_FOLDER="/home/dac/zjn_data/llama-1b-assembled-new/"

# # 循环测试不同 sparsity
# for BASE_SPARSITY in $(seq $START_SPARSITY $END_SPARSITY); do
#     echo "Running with base_sparsity=${BASE_SPARSITY} on GPU ${GPU_ID}"
    
#     GENE_PATH="${GENE_FOLDER}/${BASE_SPARSITY}-gene_info.jsonl"
    
#     # CUDA_VISIBLE_DEVICES=$GPU_ID python assemble_new.py \
#     CUDA_VISIBLE_DEVICES=$GPU_ID python assemble_attn_ffn.py \
#         --sparse_model_path "$SPARSE_MODEL_PATH" \
#         --full_model_path "$FULL_MODEL_PATH" \
#         --base_sparsity "$BASE_SPARSITY" \
#         --layer_num "$LAYER_NUM" \
#         --gene_path "$GENE_PATH" \
#         --output_folder "$OUTPUT_FOLDER"
    
#     echo "Done base_sparsity=${BASE_SPARSITY}"
#     echo "---------------------------------------------"
# done




# GPU_ID=0

# START_SPARSITY=32
# END_SPARSITY=32

# SPARSE_MODEL_PATH="/home/dac/zjn_data/Qwen3-1.7B-pruned/"
# FULL_MODEL_PATH="/home/dac/zjn_data/Qwen3-1.7B/"
# LAYER_NUM=28
# GENE_FOLDER="Qwen3-searched-3-stage-all/"
# OUTPUT_FOLDER="/home/dac/zjn_data/Qwen3-1.7B-assembled/"

# # 循环测试不同 sparsity
# for BASE_SPARSITY in $(seq $START_SPARSITY $END_SPARSITY); do
#     echo "Running with base_sparsity=${BASE_SPARSITY} on GPU ${GPU_ID}"
    
#     GENE_PATH="${GENE_FOLDER}/${BASE_SPARSITY}-gene_info.jsonl"
    
#     CUDA_VISIBLE_DEVICES=$GPU_ID python assemble_new.py \
#         --sparse_model_path "$SPARSE_MODEL_PATH" \
#         --full_model_path "$FULL_MODEL_PATH" \
#         --base_sparsity "$BASE_SPARSITY" \
#         --layer_num "$LAYER_NUM" \
#         --gene_path "$GENE_PATH" \
#         --output_folder "$OUTPUT_FOLDER"
    
#     echo "Done base_sparsity=${BASE_SPARSITY}"
#     echo "---------------------------------------------"
# done



GPU_ID=0

START_SPARSITY=32
END_SPARSITY=32

SPARSE_MODEL_PATH="/home/dac/zjn_data/opt-1.3b-pruned/"
FULL_MODEL_PATH="/home/dac/zjn_data/opt-1.3b/"
LAYER_NUM=24
# GENE_FOLDER="Qwen3-searched-3-stage-all/"
GENE_FOLDER="opt-1.3b-stage-all/"
OUTPUT_FOLDER="/home/dac/zjn_data/opt-1.3b-assembled-new/"

# 循环测试不同 sparsity
for BASE_SPARSITY in $(seq $START_SPARSITY $END_SPARSITY); do
    echo "Running with base_sparsity=${BASE_SPARSITY} on GPU ${GPU_ID}"
    
    GENE_PATH="${GENE_FOLDER}/${BASE_SPARSITY}-gene_info.jsonl"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID python assemble_opt.py \
        --sparse_model_path "$SPARSE_MODEL_PATH" \
        --full_model_path "$FULL_MODEL_PATH" \
        --base_sparsity "$BASE_SPARSITY" \
        --layer_num "$LAYER_NUM" \
        --gene_path "$GENE_PATH" \
        --output_folder "$OUTPUT_FOLDER"
    
    echo "Done base_sparsity=${BASE_SPARSITY}"
    echo "---------------------------------------------"
done

python assemble_opt.py \
    --sparse_model_path "/home/dac/zjn_data/opt-1.3b-pruned/" \
    --full_model_path "/home/dac/zjn_data/opt-1.3b/" \
    --base_sparsity 32 \
    --layer_num 24 \
    --gene_path "opt-1.3b-stage-all/32-gene_info.jsonl" \
    --output_folder "/home/dac/zjn_data/opt-1.3b-assembled-new/"