import argparse
import os 
import time 
import numpy as np
import json 
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_sparsity(model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    if "OPT" in model.__class__.__name__:
        layers = model.model.decoder.layers
    else:     
        layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()
            # print(" name ", (W==0).sum().item() / W.numel(), name, W.shape)

        # print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def get_llm(model_name, cache_dir="llm_weights"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        cache_dir=cache_dir, 
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model



parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='LLaMA model')
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4", "32:64", "33:64", "31:64"])
parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", 
                    "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter", 
                    "ablate_wanda_iter", "search", "pruner-zero", "ablate_prunerzero_seq", "ablate_prunerzero_iter"])
parser.add_argument("--cache_dir", default="llm_weights", type=str )
parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
parser.add_argument('--save', type=str, default=None, help='Path to save results.')
parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
# gradient_path 
parser.add_argument("--gradient_path", type=str, default=None, help="Path to save the gradient.")
parser.add_argument("--json_tree", type=str, default="data/best_tree.json", help="Path to load the json tree.")
parser.add_argument("--eval_zero_shot", action="store_true")
args = parser.parse_args()
    
    


print(args.model, args.cache_dir)
# /home/dac/zjn_data/opt-1.3b-pruned/00 llm_weights

for i in range(64):
    model = get_llm(args.model + f"{i:02d}", args.cache_dir)
    sparsity_ratio = check_sparsity(model)
    # assert sparsity_ratio == i / 64
    if sparsity_ratio != i / 64:
        print(f"Model {args.model + f'{i:02d}'} sparsity check failed: {sparsity_ratio:.4f} != {i/64:.4f}")

# model = get_llm(args.model, args.cache_dir)
# print("*"*30)
# sparsity_ratio = check_sparsity(model)
# print(f"sparsity sanity check {sparsity_ratio:.4f}")
# print("*"*30)