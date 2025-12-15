import warnings
warnings.filterwarnings("ignore")
import torch
import random
from qwen_rheo import Qwen2ForCausalLM_RHEO, Sparse_Linear
from transformers import Qwen2Config, AutoTokenizer
from datasets import load_dataset
import numpy as np
from torch import nn
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--sparse_path', type=str)
parser.add_argument('--base_sparsity', type=int)
parser.add_argument('--num_layers', type=int)
parser.add_argument('--sparse_model', type=str)

args = parser.parse_args()
# For reproducibility
random.seed(0)
torch.manual_seed(0)

# QWEN_PATH = "/home/dac/zjn_data/Qwen2.5-1.5B-Instruct"
# QWEN_PATH = "/home/zjnyly/repos/llmacc/rheo/Qwen2.5-1.5B-Instruct/32"
QWEN_PATH = args.model


def load_last_gene(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].strip()  # 获取最后一行并去除换行符
        data = json.loads(last_line)  # 解析 JSON
        gene = data['gene']           # 提取 gene 数组
        return gene

config = Qwen2Config.from_pretrained(QWEN_PATH)
sparse_path = load_last_gene(args.sparse_path)
# config["sparse_path"] = sparse_path
model = Qwen2ForCausalLM_RHEO.from_pretrained(
    # pretrained_model_name_or_path='meta-llama/Llama-2-7b-hf',
    pretrained_model_name_or_path=QWEN_PATH,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    # sparse_path = sparse_path
).cpu()

# state_dict = torch.load("sparse_weights.pth", map_location='cpu')
# model.load_state_dict(state_dict, strict=False)
state_dict = np.load(args.sparse_model, allow_pickle=True).item()
# print(type(state_dict))


def load_sparse_weights(model: nn.Module, compressed_state_dict: dict, sparse_path: list, num_layers: int, base_sparsity: int):
    
    sparse_path = np.array(sparse_path).reshape([-1, num_layers])
    print(sparse_path)
    sparse_path = sparse_path.transpose().flatten()
    
    
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, Sparse_Linear):
            cur_sparsity = 64 - (base_sparsity + sparse_path[cnt])
            # cur_sparsity = 32
            cnt+=1
            # print(f"Loading sparse weights for module: {name}")
            prefix = name + "."
            weight_key = prefix + "packed_weights"
            mask_key = prefix + "packed_mask"
            bias_key = prefix + "bias"
            # 加载并转换为 tensor
            packed_weights = torch.tensor(compressed_state_dict[weight_key], dtype=torch.float16)
            packed_mask = torch.tensor(compressed_state_dict[mask_key], dtype=torch.uint64)
            IC = module.IC
            OC = module.OC
            print(name, IC * OC * cur_sparsity // 64, packed_weights.shape, cur_sparsity)
            module.packed_weights = torch.empty(IC * OC * cur_sparsity // 64, dtype=torch.float16)
            # module.packed_weights = torch.empty(IC * OC * 32 // 64, dtype=torch.float16)
            module.packed_weights.copy_(packed_weights)
            module.packed_mask.copy_(packed_mask)
            module.sparsity = cur_sparsity
            # module.sparsity = 16

            if module.bias is not None and compressed_state_dict[bias_key] is not None:
                bias_tensor = torch.tensor(compressed_state_dict[bias_key], dtype=torch.float16)
                module.bias.copy_(bias_tensor)

            # output 是中间缓存，通常推理前预初始化为 0 即可
            module.output.zero_()
load_sparse_weights(model, state_dict, sparse_path, args.num_layers, args.base_sparsity)
model.cuda()
# exit()

enc = AutoTokenizer.from_pretrained(
    QWEN_PATH, 
    use_fast=False, 
    trust_remote_code=True)

# dataset = load_dataset('gsm8k', 'main')

# print("here")
# prompt = ''
# for i in range(5):
#     prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
# prompt = "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
prompt = "def fibonacci(n):"
prompt = "Question: Tell me about Elon Musk\n"
inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

import time

# ... existing code ...

inputs = enc(prompt, return_tensors="pt").input_ids.cuda()

# --- Profiling Generation ---
print("Starting generation with profiling...")
token_times = []
output_ids = inputs.clone()
past_key_values = None

model.eval()
with torch.no_grad():
    # First token (prompt processing)
    start_time = time.time()
    torch.cuda.synchronize()
    outputs = model(input_ids=output_ids, use_cache=True)
    torch.cuda.synchronize()
    end_time = time.time()
    
    token_times.append(end_time - start_time)
    print(f"Prompt processing time: {token_times[0]:.4f} seconds")

    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    output_ids = torch.cat([output_ids, next_token], dim=-1)

    # Subsequent tokens (decoding)
    for i in range(1, 200): # Generate up to 199 new tokens
        start_time = time.time()
        torch.cuda.synchronize()
        
        outputs = model(input_ids=next_token, past_key_values=past_key_values, use_cache=True)
        
        torch.cuda.synchronize()
        end_time = time.time()
        token_times.append(end_time - start_time)

        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        output_ids = torch.cat([output_ids, next_token], dim=-1)

        if next_token.item() in [151645, 151643]:
            print("EOS token found. Stopping generation.")
            break

output = output_ids
config_str = f"# prompt tokens: {inputs.shape[1]}"

print("\n--- Profiling Results ---")
print(f"Total tokens generated: {len(token_times)}")
print(f"Prompt processing time (first token): {token_times[0]:.4f} s")
if len(token_times) > 1:
    avg_decode_time = sum(token_times[1:]) / len(token_times[1:])
    tokens_per_sec = 1 / avg_decode_time
    print(f"Average decoding time per token: {avg_decode_time:.4f} s")
    print(f"Tokens per second (decoding): {tokens_per_sec:.2f} tok/s")
print("Per-token times (in seconds):", [f"{t:.4f}" for t in token_times])
print("--- End Profiling Results ---\n")

print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nRheo Output:")
print(enc.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))

# https://chatgpt.com/c/6853b2e9-1b1c-8010-a911-0e18900393d9