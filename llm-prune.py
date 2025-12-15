import argparse
import os 
import time 
import numpy as np
import json 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from prune import prune_sparsegpt, prune_magnitude, prune_wanda

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map="auto"
    )

    model.seqlen = model.config.max_position_embeddings 
    return model


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='LLaMA model')
parser.add_argument('--seed', type=int, default=42, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=512, help='Number of calibration samples.')
parser.add_argument("--sparsity_type", type=str, default="4:8")
parser.add_argument("--method", type=str, default="sparsegpt")
parser.add_argument("--base_sparsity", type=int, default=32)
parser.add_argument("--use_variant", type=bool, default=False)
parser.add_argument("--sparsity_ratio", type=float, default=0.5)
parser.add_argument("--special_tag", type=str, default="")
parser.add_argument("--folder_path", type=str, default="")




args = parser.parse_args()

# Setting seeds for reproducibility
np.random.seed(args.seed)
torch.random.manual_seed(args.seed)

prune_n, prune_m = map(int, args.sparsity_type.split(":"))

model_name = args.model.split("/")[-1]
print(f"loading llm model {args.model}")
model = get_llm(args.model)
model.eval()
# print(model)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

device = torch.device("cuda:0")


if args.method == 'sparsegpt':
    # prune_sparsegpt(args, model, tokenizer, device, prune_n, prune_m, base_sparsity=prune_n, sparsity=args.sparsity_ratio)
    prune_sparsegpt(args, model, tokenizer, device, prune_n, prune_m)
    ######################### NEED TO CHANGE BACK #########################
    # prune_sparsegpt(args, model, tokenizer, device, 0, prune_m, base_sparsity=0, sparsity=args.sparsity_ratio)
elif args.method == 'magnitude':
    prune_magnitude(args, model, tokenizer, device, base_sparsity=prune_n)
elif args.method == 'wanda':
    prune_wanda(args, model, tokenizer, device, prune_n, prune_m, base_sparsity=prune_n)





# folder_path = model_name + "/" +  "{:02d}".format(prune_n) + "-" + args.method
if args.folder_path == "":
    folder_path = "/home/dac/zjn_data/" + model_name + args.special_tag + "/" +  "{:02d}".format(prune_n)
else:
    folder_path = args.folder_path + "/" +  "{:02d}".format(prune_n)
os.makedirs(folder_path, exist_ok=True)

model.save_pretrained(folder_path)
tokenizer.save_pretrained(folder_path)
    
    
