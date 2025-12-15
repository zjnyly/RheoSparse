# python transform_weights_arbitrary_sparsity.py --model 27/ --base_sparsity 27 --sparse_path 27-gene_info.jsonl --num_layers 28
# python transform_weights_arbitrary_sparsity.py \
#     --model ~/repos/llmacc/rheo-new/EXPERIMENTS/qwen2.5/qwen2.5-3264-searched \
#     --base_sparsity 32 --sparse_path  ~/repos/llmacc/rheo-new/EXPERIMENTS/qwen2.5/qwen2.5-3264-searched/32-gene_info.jsonl --num_layers 28



python transform_weights_arbitrary_sparsity.py \
    --model 27/ \
    --base_sparsity 27 --sparse_path  27/27-gene_info.jsonl --num_layers 28

