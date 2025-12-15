import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import random
import time
import copy
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch.nn as nn
import json
import argparse
import os
import torch.nn.functional as F

# 定义参数
parser = argparse.ArgumentParser()
parser.add_argument('--full_model_path', type=str)
parser.add_argument('--sparse_model_path', type=str)
parser.add_argument('--special_tag', type=str)
parser.add_argument('--base_sparsity', type=int)
parser.add_argument('--layer_num', type=int)
parser.add_argument('--resume', type=str, help="Path to a JSON file to resume search from.")
parser.add_argument('--start', type=int, default=0, help="Starting generation for resuming search.")
args = parser.parse_args()

FULL_MODEL_PATH = args.full_model_path
SPARSE_MODEL_PATH = args.sparse_model_path
BASE_SPARSITY = args.base_sparsity
SPARSE_TYPES = {
    i: SPARSE_MODEL_PATH + "/" + f"{BASE_SPARSITY + i:02d}"
    for i in range(-32, 33)
}
LAYER_NUM = args.layer_num
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_absolute_gene(base_gene, gene, depth):
    """
    根据基础基因和当前基因生成绝对基因。
    使用新的缩放因子 (8 // (2**depth))。
    """
    absolute_gene = base_gene.copy()
    for i, g in enumerate(gene):
        absolute_gene[i] = g * (8 // (2**depth)) + base_gene[i]
    return absolute_gene

def assemble_model(model, absolute_gene, assembled_parts_cache=None):
    """
    懒加载并去重装配模型。
    """
    if assembled_parts_cache is None:
        assembled_parts_cache = {}

    # 映射基因索引到模型层名
    gene_map = [
        (0, LAYER_NUM, "self_attn.q_proj"),
        (LAYER_NUM, 2 * LAYER_NUM, "self_attn.k_proj"),
        (2 * LAYER_NUM, 3 * LAYER_NUM, "self_attn.v_proj"),
        (3 * LAYER_NUM, 4 * LAYER_NUM, "self_attn.o_proj"),
        (4 * LAYER_NUM, 5 * LAYER_NUM, "mlp.gate_proj"),
        (5 * LAYER_NUM, 6 * LAYER_NUM, "mlp.up_proj"),
        (6 * LAYER_NUM, 7 * LAYER_NUM, "mlp.down_proj"),
    ]

    # 预加载所有需要的稀疏模型路径，并进行去重

    required_sparse_types = sorted(list(set(absolute_gene)))
    print(absolute_gene)
    print(f"预加载稀疏模型: {required_sparse_types}")
    loaded_sparse_models = {}
    for g in required_sparse_types:
        if g not in SPARSE_TYPES:
            print(f"警告: 稀疏度标签 {g} 未在 SPARSE_TYPES 中定义。")
            continue

        path = SPARSE_TYPES[g]
        print(f"从 {path} 加载稀疏模型...")
        try:
            loaded_sparse_models[g] = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cpu()
        except Exception as e:
            print(f"从 {path} 加载模型失败. 错误: {e}")
            loaded_sparse_models[g] = None

    for i, g in enumerate(absolute_gene):
        layer_name = None

        for start_idx, end_idx, proj_name in gene_map:
            if start_idx <= i < end_idx:
                current_layer_idx = i - start_idx
                layer_name = f"model.layers.{current_layer_idx}.{proj_name}"
                break

        if layer_name is None:
            continue

        part_key = (layer_name, g)
        if part_key in assembled_parts_cache:
            # 如果该层的该稀疏度已装配，则跳过
            continue

        sparse_model = loaded_sparse_models.get(g)
        if sparse_model is None:
            print(f"警告: 稀疏度为 {g} 的模型未找到。跳过层 {layer_name}。")
            continue

        try:
            target_module = dict(model.named_modules())[layer_name]
            sparse_module = dict(sparse_model.named_modules())[layer_name]
            target_module.weight.data.copy_(sparse_module.weight.data)
            assembled_parts_cache[part_key] = True
        except KeyError as e:
            print(f"警告: 未找到模块 {layer_name}。错误: {e}")

    return model.to(DEVICE), assembled_parts_cache

def evaluate_ppl(full_model, sparse_model, n_samples=2048):
    """
    评估模型困惑度（PPL）。
    """
    ppl = evaluator.evaluate(full_model, sparse_model, n_samples=n_samples)
    return ppl

def crossover_and_mutate(parent1, parent2, depth, generation_depth):
    """
    遗传算法中的交叉和变异操作。
    """
    random.seed(time.time())
    if depth > 512: return random.choice([parent1, parent2])

    point = random.randint(1, LAYER_NUM - 2)
    child = parent1[:point] + parent2[point:]

    # 检查并修正平衡性
    if sum(child) != 0:
        # 如果交叉后不平衡，重新进行交叉，直到平衡
        return crossover_and_mutate(parent1, parent2, depth + 1, generation_depth)

    mutation_type = random.choice(["up", "down"])
    mutation_amount = random.randint(1, 4) * (8 // (2**generation_depth))

    attempts = 0
    max_attempts = 100 # To prevent infinite loops
    mutation_applied = False

    while not mutation_applied and attempts < max_attempts:
        attempts += 1

        # Pick two random distinct indices
        idx1, idx2 = random.sample(range(LAYER_NUM), 2)

        if mutation_type == "up":
            # 增加一个基因，减少另一个基因，保持总和不变
            if child[idx1] - mutation_amount >= -4 and child[idx2] + mutation_amount <= 4:
                child[idx1] -= mutation_amount
                child[idx2] += mutation_amount
                mutation_applied = True
        elif mutation_type == "down":
            # 减少一个基因，增加另一个基因，保持总和不变
            if child[idx1] + mutation_amount <= 4 and child[idx2] - mutation_amount >= -4:
                child[idx1] += mutation_amount
                child[idx2] -= mutation_amount
                mutation_applied = True

    return child

def generate_balanced_individual(layer_num, base_gene, depth):
    """
    生成一个平衡的个体基因。
    """
    concatenated_gene = []


    for stage in range(7):
        # 从 [-4, -3, -2, -1] 中随机选择一个值
        negative_value = random.choice([-4, -3, -2, -1])
        # 对应的正值
        positive_value = -negative_value
        gene = [0] * layer_num
        idx1, idx2 = random.sample(range(layer_num), 2)
        gene[idx1] = negative_value
        gene[idx2] = positive_value
        if(gene[idx1] + base_gene[len(concatenated_gene) + idx1] < -32 or gene[idx2] + base_gene[len(concatenated_gene) + idx2] > 32):
            gene[idx1] = 0
            gene[idx2] = 0
        for i, g in enumerate(gene):
            # print("add ", g * (8 // (2**depth)), g, 8 // (2**depth))
            gene[i] = g * (8 // (2**depth)) + base_gene[stage * layer_num + i]
        assert sum(gene) == 0, f"生成的基因不平衡: {gene}，和为 {sum(gene)}"

        concatenated_gene += gene

    return concatenated_gene

def checkBest(ppl, best_ppl, best_gene, absolute_gene):


    """
    检查并保存最佳基因。
    """
    if ppl < best_ppl:
        best_ppl = ppl
        best_gene = absolute_gene
        info = {
            "gene": best_gene,
            "ppl": ppl.item() if isinstance(ppl, torch.Tensor) else ppl,
        }
        write_path = "Qwen2.5_attn_ffn/"
        os.makedirs(write_path, exist_ok=True)
        with open(write_path + str(args.base_sparsity) + "-gene_info-new.jsonl", "a") as f:
            f.write(json.dumps(info) + "\n")
        print(f"✅ 新的最佳 PPL={ppl:.4f}")
    return best_ppl, best_gene




def genetic_search(full_model, base_gene, depth, population, generations=1000, population_size=64, best_ppl=10000):
    """
    主遗传搜索循环。
    """
    best_gene = None
    model = copy.deepcopy(full_model)
    assembled_parts_cache = {}

    for gen in range(generations):
        print(f"\n--- 第 {gen} 代 ---，种群大小为 {population_size}")
        scores = []

        for i, gene in enumerate(population):
            # absolute_gene = generate_absolute_gene(base_gene, gene, depth)
            absolute_gene = gene

            model, assembled_parts_cache = assemble_model(model, absolute_gene, assembled_parts_cache)

            ppl = evaluate_ppl(full_model, model, 2048)
            print(f"{i}, PPL: {ppl}, 目前最佳 PPL: {best_ppl}")
            best_ppl, best_gene = checkBest(ppl, best_ppl, best_gene, absolute_gene)
            scores.append((ppl, gene))

        scores.sort()
        parents = [gene for _, gene in scores[:2]]

        if len(parents) < 2: continue
        new_population = []
        new_population.append(parents[0])
        new_population.append(parents[1])
        while len(new_population) < population_size:
            p1, p2 = random.sample(parents, 2)
            child = []
            for part in range(7):
                child += crossover_and_mutate(p1[LAYER_NUM * part:LAYER_NUM * (part + 1)], p2[LAYER_NUM * part :LAYER_NUM * (part + 1)], 0, depth)
            new_population.append(child)

        population = new_population

    print("\n=== 搜索完成 ===")
    print("最佳基因:", best_gene)
    print("最佳 PPL:", best_ppl)
    return best_gene, best_ppl

class Evaluator:
    """
    评估器类，用于计算模型的KL散度。
    """
    def __init__(self, dataset):
        self.dataset = dataset

    @torch.inference_mode()
    def evaluate(self, full_model, sparse_model, n_samples=1, batch_size=2):
        full_model.eval()
        sparse_model.eval()

        global_mean_result = 0.0
        total_processed = 0

        # 分批处理，每批最多 batch_size 个样本
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_samples = []

            # 收集当前批次样本（跳过超长样本）
            for idx in range(start_idx, end_idx):
                sample = self.dataset[idx]
                if sample.shape[1] > 2048:
                    continue
                batch_samples.append(sample)

            if len(batch_samples) == 0:
                continue

            # 填充到相同长度（右填充）
            max_len = max(s.size(1) for s in batch_samples)
            padded_samples = []
            for s in batch_samples:
                if s.size(1) < max_len:
                    pad_len = max_len - s.size(1)
                    s = torch.cat([s, torch.full((1, pad_len), tokenizer.pad_token_id, dtype=s.dtype, device=s.device)], dim=1)
                padded_samples.append(s)

            # 合并为一个 batch
            batch_input = torch.cat(padded_samples, dim=0).cuda()  # [B, L]

            # 前向传播
            result_full = full_model(batch_input, output_hidden_states=False)
            result_sparse = sparse_model(batch_input, output_hidden_states=False)

            logits_full = result_full.logits[:, :-1, :].contiguous().float()  # [B, L-1, V]
            logits_sparse = result_sparse.logits[:, :-1, :].contiguous().float()  # [B, L-1, V]

            # 使用 reduction='none' 得到每个位置的 KL
            kl_per_token = F.kl_div(
                F.log_softmax(logits_sparse, dim=-1),
                F.softmax(logits_full, dim=-1),
                reduction='none'
            ).sum(dim=-1)  # [B, L-1]，对 vocab 维度求和

            # 对每个样本，计算平均 KL（模拟 batchmean for batch_size=1）
            for i in range(len(batch_samples)):
                orig_len = batch_samples[i].size(1) - 1  # 因为 logits 是 [:, :-1, :]
                # 只取非填充部分
                sample_kl = kl_per_token[i, :orig_len].sum().item()  # 平均到每个 token
                global_mean_result += sample_kl

            total_processed += len(batch_samples)

            # 清理显存
            del result_full, result_sparse, logits_full, logits_sparse, kl_per_token, batch_input
            torch.cuda.empty_cache()

        return global_mean_result
    # @torch.no_grad()
    # @torch.inference_mode()
    # def evaluate(self, full_model, sparse_model, n_samples=1):
    #     full_model.eval()
    #     sparse_model.eval()

    #     global_mean_result = 0.0
    #     for i in range(n_samples // 256):
    #         local_mean_result = 0.0
    #         cnt = 0
    #         for j in range(256):
    #             idx = i * 256 + j
    #             sample = self.dataset[idx].cuda()
    #             if sample.shape[1] > 2048:
    #                 continue
    #             # sample = sample[:, :256]
    #             cnt += 1

    #             result_full = full_model(sample, output_hidden_states=False)

    #             result_sparse = sparse_model(sample, output_hidden_states=False)

    #             logits_full = result_full.logits[:, :-1, :].contiguous().float()
    #             logits_sparse = result_sparse.logits[:, :-1, :].contiguous().float()
    #             kl_divergence = F.kl_div(
    #                 F.log_softmax(logits_sparse, dim=-1),
    #                 F.softmax(logits_full, dim=-1),
    #                 reduction="batchmean"
    #             )

    #             del result_full, result_sparse, logits_full, logits_sparse
    #             local_mean_result += kl_divergence.item()

    #         global_mean_result += local_mean_result

    #     return global_mean_result

def get_ultrachat(nsamples, seed, seqlen, tokenizer):
    """
    获取数据集。
    """
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=161).select(range(128 * 1024))
    def preprocess(example):
        return {"text": example["messages"][0]['content']}
    ds = ds.map(preprocess)
    random.seed(seed)
    samples = []
    for i in tqdm(range(nsamples)):
        tokenized = tokenizer(ds[i]["text"], return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized.input_ids[0]
        samples.append(input_ids.unsqueeze(0))
    return samples, None

# 主执行逻辑
tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_PATH)

samples = torch.load('samples_16k.pt')[0]
full_model = AutoModelForCausalLM.from_pretrained(FULL_MODEL_PATH, torch_dtype=torch.float16).cuda()
full_model.eval()

evaluator = Evaluator(samples)

base_gene = [0] * (LAYER_NUM * 7)


def load_best_gene_from_file(file_path):
    """Loads the best gene and its PPL from a JSONL file."""
    best_ppl = float('inf')
    best_gene = None
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                if data["ppl"] < best_ppl:
                    best_ppl = data["ppl"]
                    best_gene = data["gene"]
        if best_gene:
            print(f"✅ Successfully resumed from {file_path}")
            print(f"   Initial best PPL: {best_ppl:.4f}")
            print(f"   Initial base gene: {best_gene}")
        return best_gene, best_ppl
    except FileNotFoundError:
        print(f"⚠️ Warning: Resume file '{file_path}' not found. Starting a new search.")
        return None, float('inf')
    except json.JSONDecodeError:
        print(f"❌ Error decoding JSON in {file_path}. Starting a new search.")
        return None, float('inf')

if args.resume:
    initial_gene, initial_ppl = load_best_gene_from_file(args.resume)
    if initial_gene and initial_ppl < float('inf'):
        base_gene = initial_gene
        best_ppl_pev = initial_ppl


best_ppl_pev = 1e9
for i in range(args.start,  4):
    print(f"开始第 {i} 次迭代")
    population = [generate_balanced_individual(layer_num=LAYER_NUM, base_gene=base_gene, depth=i) for _ in range(128)]
    best_gene, best_ppl = genetic_search(full_model, base_gene, depth = i, population = population, generations=2, population_size=128, best_ppl = best_ppl_pev)

    best_ppl_pev = best_ppl
    if best_gene is None:
        best_gene = [0] * (7 * LAYER_NUM)
    base_gene = best_gene
    print(f"最佳 PPL {best_ppl}")