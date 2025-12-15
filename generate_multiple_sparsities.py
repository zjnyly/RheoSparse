import os

model_path = "/home/dac/zjn_data/Qwen2.5-1.5B-Instruct"

script_path = "llm-prune.py"

method = 'sparsegpt'  # or 'magnitude', 'wanda'

output_folder = "/home/dac/zjn_data/Qwen2.5-1.5B-Instruct-pruned/"

sparsity_values = list(range(1, 64)) 

for value in sparsity_values:
    sparsity_type = f"{value}:64"
    command = f"python {script_path} --model {model_path} --sparsity_type {sparsity_type} --method {method} --folder_path {output_folder} --sparsity_ratio {value/64}"
    print(f"Running: {command}")
    os.system(command)
    
    