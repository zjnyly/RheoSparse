# import torch
# import random
# import time
# import copy
# import numpy as np
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from datasets import load_dataset
# import tqdm
# import torch
# import torch.nn as nn
# import json
# import argparse
# import os
# import torch.nn.functional as F

# parser = argparse.ArgumentParser()
# parser.add_argument('--full_model_path', type=str)
# parser.add_argument('--sparse_model_path', type=str)
# parser.add_argument('--special_tag', type=str)
# parser.add_argument('--base_sparsity', type=int)
# parser.add_argument('--layer_num', type=int)
# args = parser.parse_args()

# FULL_MODEL_PATH = args.full_model_path
# SPARSE_MODEL_PATH = args.sparse_model_path
# BASE_SPARSITY = args.base_sparsity
# SPARSE_TYPES = {
#     -7: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 7),
#     -6: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 6),
#     -5: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 5),
#     -4: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 4),
#     -3: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 3),
#     -2: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 2),
#     -1: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 1),
#     0: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 0),
#     1: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 1),
#     2: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 2),
#     3: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 3),
#     4: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 4),
#     5: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 5),
#     6: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 6),
#     7: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 7),
# }
# LAYER_NUM = args.layer_num 
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# def load_sparse_models():
#     models = {}
#     for k, path in SPARSE_TYPES.items():
#         print(f"Loading sparse model from {path}")
#         models[k] = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cpu()
#     return models

# def generate_absolute_gene(base_gene, gene, depth):
#     absolute_gene = base_gene.copy()
#     for i, g in enumerate(gene):
#         absolute_gene[i] = g * (4 // (2**depth)) + base_gene[i]
#     return absolute_gene
    

# def assemble_model(models, absolute_gene):
#     model = copy.deepcopy(models[0]) 
    
#     for i, g in enumerate(absolute_gene):
#         # g = g * (2 // (2**depth)) + base_gene[i]
#         # print(g, " ", end = "")
#         if i < LAYER_NUM:
#             layer_name = f"model.layers.{i}.self_attn.q_proj"
#             target_module = dict(model.named_modules())[layer_name]
#             sparse_module = dict(models[0].named_modules())[layer_name]
#             target_module.weight.data.copy_(sparse_module.weight.data)
#         elif i < 2 * LAYER_NUM:
#             layer_name = f"model.layers.{i-LAYER_NUM}.self_attn.k_proj"
#             target_module = dict(model.named_modules())[layer_name]
#             sparse_module = dict(models[0].named_modules())[layer_name]
#             target_module.weight.data.copy_(sparse_module.weight.data)
#         elif i < 3 * LAYER_NUM:
#             layer_name = f"model.layers.{i-LAYER_NUM*2}.self_attn.v_proj"
#             target_module = dict(model.named_modules())[layer_name]
#             sparse_module = dict(models[0].named_modules())[layer_name]
#             target_module.weight.data.copy_(sparse_module.weight.data)
#         elif i < 4 * LAYER_NUM:
#             layer_name = f"model.layers.{i-LAYER_NUM*3}.self_attn.o_proj"
#             target_module = dict(model.named_modules())[layer_name]
#             sparse_module = dict(models[0].named_modules())[layer_name]
#             target_module.weight.data.copy_(sparse_module.weight.data)
#         elif i < 5 * LAYER_NUM:
#             layer_name = f"model.layers.{i-LAYER_NUM*4}.mlp.gate_proj"
#             target_module = dict(model.named_modules())[layer_name]
#             sparse_module = dict(models[g].named_modules())[layer_name]
#             target_module.weight.data.copy_(sparse_module.weight.data)
#         elif i < 6 * LAYER_NUM:
#             layer_name = f"model.layers.{i-LAYER_NUM*5}.mlp.up_proj"
#             target_module = dict(model.named_modules())[layer_name]
#             sparse_module = dict(models[0].named_modules())[layer_name]
#             target_module.weight.data.copy_(sparse_module.weight.data)
#         else:
#             layer_name = f"model.layers.{i-LAYER_NUM*6}.mlp.down_proj"
#             target_module = dict(model.named_modules())[layer_name]
#             sparse_module = dict(models[0].named_modules())[layer_name]
#             target_module.weight.data.copy_(sparse_module.weight.data)
#     return model.to(DEVICE)


# def evaluate_ppl(full_model, sparse_model, n_samples=2048):
#     ppl = evaluator.evaluate(full_model, sparse_model, n_samples=n_samples)
    
#     return ppl


# def crossover_and_mutate(parent1, parent2, depth):
#     random.seed(time.time())
#     if depth > 512: return random.choice([parent1, parent2])
    
#     point = random.randint(1, LAYER_NUM - 2)
#     child = parent1[:point] + parent2[point:]

#     if child.count(0) != child.count(2):
#         return crossover_and_mutate(parent1, parent2, depth + 1)
#     mutation_type = random.choice(["up", "down"])

#     if mutation_type == "up":
#         indices = [i for i in range(LAYER_NUM) if child[i] == 1]
#         if len(indices) >= 2:
#             chosen_indices = random.sample(indices, 2)
#             child[chosen_indices[0]] = 0 
#             child[chosen_indices[1]] = 2 
#     elif mutation_type == "down":
#         indices_0 = [i for i in range(LAYER_NUM) if child[i] == 0]
#         indices_2 = [i for i in range(LAYER_NUM) if child[i] == 2]
        
#         if indices_0 and indices_2:
#             chosen_0 = random.choice(indices_0)
#             chosen_2 = random.choice(indices_2)
            
#             child[chosen_0] = 1
#             child[chosen_2] = 1

#     return child

# def generate_balanced_individual(layer_num, shuffle):
#     assert layer_num % 2 == 0, "layer_num 必须是偶数才能平衡 0 和 2"
#     concatenated_gene = []
#     for i in range(7):
#         gene = [0] * layer_num
#         idx1, idx2 = random.sample(range(layer_num), 2)
#         gene[idx1] = -1
#         gene[idx2] = 1
#         concatenated_gene += gene
    
#     return concatenated_gene

# def is_balanced_02(individual):
#     return individual.count(-1) == individual.count(1)

# def checkBest(ppl, best_ppl, best_gene, absolute_gene):
#     if ppl < best_ppl:
#         best_ppl = ppl
#         # model.save_pretrained(args.sparse_model_path)
#         # tokenizer.save_pretrained(args.sparse_model_path)
#         best_gene = absolute_gene
#         info = {
#             "gene": best_gene,
#             # "sparsity_ratio": sum(x in [0, 2] for x in best_gene) / len(best_gene),
#             "ppl": ppl.item() if isinstance(ppl, torch.Tensor) else ppl,
            
#         }
#         os.makedirs("llama-1b-searched-3-stage-all/", exist_ok=True)
#         with open("llama-1b-searched-3-stage-all/" + str(args.base_sparsity) + "-gene_info.jsonl", "a") as f:
#             f.write(json.dumps(info) + "\n")
#         print(f"✅ New best PPL={ppl:.4f}")
#     return best_ppl, best_gene


# def genetic_search( models, full_model, base_gene, depth, population, generations=1000, population_size=64, best_ppl=10000):
#     best_gene = None
#     for gen in range(generations):
#         print(f"\n--- Generation {gen} ---, with population size {population_size}")
#         prev_gene = None
#         scores = []
        
#         # print(" Stage 1 ")
#         for i, gene in enumerate(population):
            
#             assert is_balanced_02(gene) == True
#             absolute_gene = generate_absolute_gene(base_gene, gene, depth)
#             if prev_gene is not None and prev_gene == absolute_gene:
#                 continue
#             prev_gene = absolute_gene
#             model = assemble_model(models, absolute_gene)
#             ppl = evaluate_ppl(full_model, model, 256)
#             print(i, ppl, best_ppl)
#             best_ppl, best_gene = checkBest(ppl, best_ppl, best_gene, absolute_gene)
#             scores.append((ppl, gene))
    
    
#         # scores.sort()
#         # parents = [gene for _, gene in scores[:16]]
#         # scores = []
#         # print(" Stage 2 ")
#         # for i, gene in enumerate(parents):
#         #     print(i)
#         #     assert is_balanced_02(gene) == True
#         #     absolute_gene = generate_absolute_gene(base_gene, gene, depth)
#         #     if prev_gene is not None and prev_gene == absolute_gene:
#         #         continue
#         #     prev_gene = absolute_gene
#         #     model = assemble_model(models, absolute_gene)
#         #     ppl = evaluate_ppl(full_model, model, 300)
#         #     best_ppl = checkBest(ppl, best_ppl, absolute_gene)
#         #     scores.append((ppl, gene))
        
        
#         # scores.sort()
#         # parents = [gene for _, gene in scores[:4]]
#         # scores = []
#         # print(" Stage 3 ")
#         # for i, gene in enumerate(parents):
#         #     print(i)
#         #     assert is_balanced_02(gene) == True
#         #     absolute_gene = generate_absolute_gene(base_gene, gene, depth)
#         #     if prev_gene is not None and prev_gene == absolute_gene:
#         #         continue
#         #     prev_gene = absolute_gene
#         #     model = assemble_model(models, absolute_gene)
#         #     ppl = evaluate_ppl(full_model, model, 400)
#         #     scores.append((ppl, gene))
#         #     best_ppl = checkBest(ppl, best_ppl, absolute_gene)
        
        
#         scores.sort()
#         parents = [gene for _, gene in scores[:2]]
        
#         if len(parents) < 2: continue
#         new_population = []
#         new_population.append(parents[0])
#         new_population.append(parents[1])
#         while len(new_population) < population_size:
#             p1, p2 = random.sample(parents, 2)
#             child = []
#             for part in range(7):
#                 child += crossover_and_mutate(p1[LAYER_NUM * part:LAYER_NUM * (part + 1)], p2[LAYER_NUM * part :LAYER_NUM * (part + 1)], 0)
#             new_population.append(child)

#         population = new_population
#         # population_size /= 4

#     print("\n=== Search Done ===")
#     print("Best Gene:", best_gene)
#     print("Best PPL:", best_ppl)
    
        
#     return best_gene, best_ppl

# class Evaluator:
#     def __init__(self, dataset):
#         self.dataset = dataset
       
#     @torch.no_grad()
#     def evaluate(self, full_model, sparse_model, n_samples=2048):
#         full_model.eval()
#         sparse_model.eval()
        
#         alpha = 0.5  # KL divergence weight
#         beta = 0.5  # L2 distance weight
        
#         global_mean_result = 0.0
#         for i in range(n_samples // 256):
#             local_mean_result = 0.0
#             cnt = 0
#             for j in range(256):
#                 idx = i * 256 + j
                
#                 sample = self.dataset[idx].cuda()
#                 if sample.shape[1] > 2048:
#                     continue
#                 cnt += 1
#                 # print(sample.shape[1])
                
#                 result_full = full_model(sample, output_hidden_states=True)
#                 result_sparse = sparse_model(sample, output_hidden_states=True)
               
#                 l2_distance = sum(
#                     F.mse_loss(f, s, reduction='mean')  # 每层按 token 平均
#                     for f, s in zip(result_full.hidden_states, result_sparse.hidden_states)
#                 ) 
#                 logits_full = result_full.logits[:, :-1, :].contiguous().float()
#                 logits_sparse = result_sparse.logits[:, :-1, :].contiguous().float()
#                 kl_divergence = F.kl_div(
#                     F.log_softmax(logits_sparse, dim=-1),
#                     F.softmax(logits_full, dim=-1),
#                     reduction="batchmean"
#                 ) / sample.shape[1]
                
#                 del result_full, result_sparse, logits_full, logits_sparse
#                 local_mean_result += alpha * kl_divergence.item() + beta * l2_distance.item()
            
#             global_mean_result += local_mean_result / cnt
#         global_mean_result /= (n_samples // 256)
#         # print(f"Global Mean Result: {global_mean_result}")
#         return global_mean_result
            




# def get_ultrachat(nsamples, seed, seqlen, tokenizer):
#     ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
#     ds = ds.shuffle(seed=161).select(range(128 * 1024))
#     def preprocess(example):
#         return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
#     ds = ds.map(preprocess)
#     random.seed(seed)
#     samples = []
#     for i in tqdm(range(nsamples)):
#         tokenized = tokenizer(ds[i]["text"], return_tensors="pt", add_special_tokens=False)
#         input_ids = tokenized.input_ids[0]
#         samples.append(input_ids.unsqueeze(0))
#     return samples, None



# tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_PATH)

# samples = torch.load('samples_16k.pt')[0]
# full_model = AutoModelForCausalLM.from_pretrained(FULL_MODEL_PATH, torch_dtype=torch.float16).cuda()
# full_model.eval()


# evaluator = Evaluator(samples)
# # evaluator.evaluate(full_model, sparse_model, n_samples=2048)

# models = load_sparse_models()

# # print(models[0].state_dict())



# base_gene = [0] * (LAYER_NUM * 7)    
# best_ppl_pev = 10000
# for i in range(3):
#     print(f"Begin iteration {i}")
#     population = [generate_balanced_individual(layer_num=LAYER_NUM, shuffle = True) for _ in range(32)]
#     population[0] = [0] * (7 * LAYER_NUM)
#     best_gene, best_ppl = genetic_search(models, full_model, base_gene, depth = i, population = population, generations=4, population_size=32, best_ppl = best_ppl_pev)
#     # base_gene = [x * 2 for x in best_gene_unbiased]
#     best_ppl_pev = best_ppl
#     if best_gene is None:
#         best_gene = [0] * (7 * LAYER_NUM)
#     base_gene = best_gene
#     print(f"best_ppl {best_ppl}")




import torch
import random
import time
import copy
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import tqdm
import torch
import torch.nn as nn
import json
import argparse
import os
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--full_model_path', type=str)
parser.add_argument('--sparse_model_path', type=str)
parser.add_argument('--special_tag', type=str)
parser.add_argument('--base_sparsity', type=int)
parser.add_argument('--layer_num', type=int)
args = parser.parse_args()

FULL_MODEL_PATH = args.full_model_path
SPARSE_MODEL_PATH = args.sparse_model_path
BASE_SPARSITY = args.base_sparsity
SPARSE_TYPES = {
    -7: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 7),
    -6: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 6),
    -5: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 5),
    -4: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 4),
    -3: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 3),
    -2: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 2),
    -1: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 1),
    0: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY - 0),
    1: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 1),
    2: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 2),
    3: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 3),
    4: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 4),
    5: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 5),
    6: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 6),
    7: SPARSE_MODEL_PATH + "/" + str(BASE_SPARSITY + 7),
}
LAYER_NUM = args.layer_num 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_sparse_models():
    models = {}
    for k, path in SPARSE_TYPES.items():
        print(f"Loading sparse model from {path}")
        models[k] = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cpu()
    return models

def generate_absolute_gene(base_gene, gene, depth):
    absolute_gene = base_gene.copy()
    for i, g in enumerate(gene):
        absolute_gene[i] = g * (4 // (2**depth)) + base_gene[i]
    return absolute_gene
    

def assemble_model(models, absolute_gene):
    model = copy.deepcopy(models[0]) 
    
    for i, g in enumerate(absolute_gene):
        # g = g * (2 // (2**depth)) + base_gene[i]
        # print(g, " ", end = "")
        if i < LAYER_NUM:
            layer_name = f"model.layers.{i}.self_attn.q_proj"
            target_module = dict(model.named_modules())[layer_name]
            sparse_module = dict(models[g].named_modules())[layer_name]
            target_module.weight.data.copy_(sparse_module.weight.data)
        elif i < 2 * LAYER_NUM:
            layer_name = f"model.layers.{i-LAYER_NUM}.self_attn.k_proj"
            target_module = dict(model.named_modules())[layer_name]
            sparse_module = dict(models[g].named_modules())[layer_name]
            target_module.weight.data.copy_(sparse_module.weight.data)
        elif i < 3 * LAYER_NUM:
            layer_name = f"model.layers.{i-LAYER_NUM*2}.self_attn.v_proj"
            target_module = dict(model.named_modules())[layer_name]
            sparse_module = dict(models[g].named_modules())[layer_name]
            target_module.weight.data.copy_(sparse_module.weight.data)
        elif i < 4 * LAYER_NUM:
            layer_name = f"model.layers.{i-LAYER_NUM*3}.self_attn.o_proj"
            target_module = dict(model.named_modules())[layer_name]
            sparse_module = dict(models[g].named_modules())[layer_name]
            target_module.weight.data.copy_(sparse_module.weight.data)
        elif i < 5 * LAYER_NUM:
            layer_name = f"model.layers.{i-LAYER_NUM*4}.mlp.gate_proj"
            target_module = dict(model.named_modules())[layer_name]
            sparse_module = dict(models[g].named_modules())[layer_name]
            target_module.weight.data.copy_(sparse_module.weight.data)
        elif i < 6 * LAYER_NUM:
            layer_name = f"model.layers.{i-LAYER_NUM*5}.mlp.up_proj"
            target_module = dict(model.named_modules())[layer_name]
            sparse_module = dict(models[g].named_modules())[layer_name]
            target_module.weight.data.copy_(sparse_module.weight.data)
        else:
            layer_name = f"model.layers.{i-LAYER_NUM*6}.mlp.down_proj"
            target_module = dict(model.named_modules())[layer_name]
            sparse_module = dict(models[g].named_modules())[layer_name]
            target_module.weight.data.copy_(sparse_module.weight.data)
    return model.to(DEVICE)


def evaluate_ppl(full_model, sparse_model, n_samples=2048):
    ppl = evaluator.evaluate(full_model, sparse_model, n_samples=n_samples)
    
    return ppl


def crossover_and_mutate(parent1, parent2, depth):
    random.seed(time.time())
    if depth > 512: return random.choice([parent1, parent2])
    
    point = random.randint(1, LAYER_NUM - 2)
    child = parent1[:point] + parent2[point:]

    if child.count(0) != child.count(2):
        return crossover_and_mutate(parent1, parent2, depth + 1)
    mutation_type = random.choice(["up", "down"])

    if mutation_type == "up":
        indices = [i for i in range(LAYER_NUM) if child[i] == 1]
        if len(indices) >= 2:
            chosen_indices = random.sample(indices, 2)
            child[chosen_indices[0]] = 0 
            child[chosen_indices[1]] = 2 
    elif mutation_type == "down":
        indices_0 = [i for i in range(LAYER_NUM) if child[i] == 0]
        indices_2 = [i for i in range(LAYER_NUM) if child[i] == 2]
        
        if indices_0 and indices_2:
            chosen_0 = random.choice(indices_0)
            chosen_2 = random.choice(indices_2)
            
            child[chosen_0] = 1
            child[chosen_2] = 1

    return child

def generate_balanced_individual(layer_num, shuffle):
    assert layer_num % 2 == 0, "layer_num 必须是偶数才能平衡 0 和 2"
    concatenated_gene = []
    for i in range(7):
        gene = [0] * layer_num
        idx1, idx2 = random.sample(range(layer_num), 2)
        gene[idx1] = -1
        gene[idx2] = 1
        concatenated_gene += gene
    
    return concatenated_gene

def is_balanced_02(individual):
    return individual.count(-1) == individual.count(1)

def checkBest(ppl, best_ppl, best_gene, absolute_gene):
    if ppl < best_ppl:
        best_ppl = ppl
        # model.save_pretrained(args.sparse_model_path)
        # tokenizer.save_pretrained(args.sparse_model_path)
        best_gene = absolute_gene
        info = {
            "gene": best_gene,
            # "sparsity_ratio": sum(x in [0, 2] for x in best_gene) / len(best_gene),
            "ppl": ppl.item() if isinstance(ppl, torch.Tensor) else ppl,
            
        }
        # write_path = "llama-1b-searched-3-stage-all/"
        write_path = "Qwen3-searched-3-stage-all/"
        os.makedirs(write_path, exist_ok=True)
        with open(write_path + str(args.base_sparsity) + "-gene_info.jsonl", "a") as f:
            f.write(json.dumps(info) + "\n")
        print(f"✅ New best PPL={ppl:.4f}")
    return best_ppl, best_gene


def genetic_search( models, full_model, base_gene, depth, population, generations=1000, population_size=64, best_ppl=10000):
    best_gene = None
    for gen in range(generations):
        print(f"\n--- Generation {gen} ---, with population size {population_size}")
        prev_gene = None
        scores = []
        
        # print(" Stage 1 ")
        for i, gene in enumerate(population):
            
            assert is_balanced_02(gene) == True
            absolute_gene = generate_absolute_gene(base_gene, gene, depth)
            if prev_gene is not None and prev_gene == absolute_gene:
                continue
            prev_gene = absolute_gene
            model = assemble_model(models, absolute_gene)
            ppl = evaluate_ppl(full_model, model, 2048)
            print(i, ppl, best_ppl)
            best_ppl, best_gene = checkBest(ppl, best_ppl, best_gene, absolute_gene)
            scores.append((ppl, gene))
    
    
        # scores.sort()
        # parents = [gene for _, gene in scores[:16]]
        # scores = []
        # print(" Stage 2 ")
        # for i, gene in enumerate(parents):
        #     print(i)
        #     assert is_balanced_02(gene) == True
        #     absolute_gene = generate_absolute_gene(base_gene, gene, depth)
        #     if prev_gene is not None and prev_gene == absolute_gene:
        #         continue
        #     prev_gene = absolute_gene
        #     model = assemble_model(models, absolute_gene)
        #     ppl = evaluate_ppl(full_model, model, 300)
        #     best_ppl = checkBest(ppl, best_ppl, absolute_gene)
        #     scores.append((ppl, gene))
        
        
        # scores.sort()
        # parents = [gene for _, gene in scores[:4]]
        # scores = []
        # print(" Stage 3 ")
        # for i, gene in enumerate(parents):
        #     print(i)
        #     assert is_balanced_02(gene) == True
        #     absolute_gene = generate_absolute_gene(base_gene, gene, depth)
        #     if prev_gene is not None and prev_gene == absolute_gene:
        #         continue
        #     prev_gene = absolute_gene
        #     model = assemble_model(models, absolute_gene)
        #     ppl = evaluate_ppl(full_model, model, 400)
        #     scores.append((ppl, gene))
        #     best_ppl = checkBest(ppl, best_ppl, absolute_gene)
        
        
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
                child += crossover_and_mutate(p1[LAYER_NUM * part:LAYER_NUM * (part + 1)], p2[LAYER_NUM * part :LAYER_NUM * (part + 1)], 0)
            new_population.append(child)

        population = new_population
        # population_size /= 4

    print("\n=== Search Done ===")
    print("Best Gene:", best_gene)
    print("Best PPL:", best_ppl)
    
        
    return best_gene, best_ppl

class Evaluator:
    def __init__(self, dataset):
        self.dataset = dataset
       
    @torch.no_grad()
    def evaluate(self, full_model, sparse_model, n_samples=2048):
        full_model.eval()
        sparse_model.eval()
        
        alpha = 0.5  # KL divergence weight
        beta = 0.5  # L2 distance weight
        
        global_mean_result = 0.0
        for i in range(n_samples // 256):
            local_mean_result = 0.0
            cnt = 0
            for j in range(256):
                idx = i * 256 + j
                
                sample = self.dataset[idx].cuda()
                if sample.shape[1] > 2048:
                    continue
                cnt += 1
                # print(sample.shape[1])
                
                result_full = full_model(sample, output_hidden_states=False)
                result_sparse = sparse_model(sample, output_hidden_states=False)
              
                logits_full = result_full.logits[:, :-1, :].contiguous().float()
                logits_sparse = result_sparse.logits[:, :-1, :].contiguous().float()
                kl_divergence = F.kl_div(
                    F.log_softmax(logits_sparse, dim=-1),
                    F.softmax(logits_full, dim=-1),
                    reduction="batchmean"
                ) 
                
                del result_full, result_sparse, logits_full, logits_sparse
                local_mean_result += kl_divergence.item() 
                # print(kl_divergence)
            
            global_mean_result += local_mean_result 
        # global_mean_result /= (n_samples // 256)
        # print(f"Global Mean Result: {global_mean_result}")
        return global_mean_result
            




def get_ultrachat(nsamples, seed, seqlen, tokenizer):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    ds = ds.shuffle(seed=161).select(range(128 * 1024))
    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}
    ds = ds.map(preprocess)
    random.seed(seed)
    samples = []
    for i in tqdm(range(nsamples)):
        tokenized = tokenizer(ds[i]["text"], return_tensors="pt", add_special_tokens=False)
        input_ids = tokenized.input_ids[0]
        samples.append(input_ids.unsqueeze(0))
    return samples, None



tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_PATH)

samples = torch.load('samples_16k.pt')[0]
full_model = AutoModelForCausalLM.from_pretrained(FULL_MODEL_PATH, torch_dtype=torch.float16).cuda()
full_model.eval()


evaluator = Evaluator(samples)
# evaluator.evaluate(full_model, sparse_model, n_samples=2048)

models = load_sparse_models()

# print(models[0].state_dict())



base_gene = [0] * (LAYER_NUM * 7)    
best_ppl_pev = 1e9
for i in range(3):
    print(f"Begin iteration {i}")
    population = [generate_balanced_individual(layer_num=LAYER_NUM, shuffle = True) for _ in range(128)]
    population[0] = [0] * (7 * LAYER_NUM)
    best_gene, best_ppl = genetic_search(models, full_model, base_gene, depth = i, population = population, generations=2, population_size=128, best_ppl = best_ppl_pev)
    # base_gene = [x * 2 for x in best_gene_unbiased]
    best_ppl_pev = best_ppl
    if best_gene is None:
        best_gene = [0] * (7 * LAYER_NUM)
    base_gene = best_gene
    print(f"best_ppl {best_ppl}")

