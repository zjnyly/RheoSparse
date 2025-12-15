import subprocess

full_model_path = "./qwen2.5"
sparse_model_path = "./qwen2.5-pruned/"
layer_num = 28

for base_sparsity in range(32, 33):  # Python range 上界不含
    cmd = [
        "python", "genetic_search_qwen_widerange.py",
        "--full_model_path", full_model_path,
        "--sparse_model_path", sparse_model_path,
        "--base_sparsity", str(base_sparsity),
        "--layer_num", str(layer_num),
        "--special_tag", "",
        # "--resume", "opt-1.3b-stage-all/32-gene_info.jsonl",
        "--start", "1"
    ]
    
    print(f"Running with base_sparsity = {base_sparsity}")
    subprocess.run(cmd)
    