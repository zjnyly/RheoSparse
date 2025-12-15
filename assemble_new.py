
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

parser = argparse.ArgumentParser()
parser.add_argument('--sparse_model_path', type=str)
parser.add_argument('--full_model_path', type=str)
parser.add_argument('--base_sparsity', type=int)
parser.add_argument('--layer_num', type=int)
parser.add_argument('--gene_path', type=str)
parser.add_argument('--output_folder', type=str)
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
        print(f"Loading sparse model {k} from {path}")
        models[k] = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16).cpu()
    return models

def generate_absolute_gene(base_gene, gene, depth):
    absolute_gene = base_gene.copy()
    for i, g in enumerate(gene):
        absolute_gene[i] = g * (2 // (2**depth)) + base_gene[i]
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

class Evaluator:
    def __init__(self, dataset, tokenizer, device, n_samples=40):
        self.dataset = dataset
        print(len(self.dataset))
        self.tokenizer = tokenizer
        self.device = device

        self.dataset = tokenizer(
            "\n\n".join(dataset["text"]), return_tensors="pt"
        ).input_ids.to(device)

        self.n_samples = n_samples

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        nlls = []
        n_samples = self.n_samples if self.n_samples else self.dataset.size(1) // 2048
        for i in tqdm.tqdm(range(n_samples), desc="Evaluating..."):
            batch = self.dataset[:, (i * 2048) : ((i + 1) * 2048)].to(model.device)
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = self.dataset[:, (i * 2048) : ((i + 1) * 2048)][:, 1:]
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            neg_log_likelihood = loss.float() * 2048
            nlls.append(neg_log_likelihood)

        return torch.exp(torch.stack(nlls).sum() / (n_samples * 2048))


def load_last_gene(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].strip()  # 获取最后一行并去除换行符
        data = json.loads(last_line)  # 解析 JSON
        gene = data['gene']           # 提取 gene 数组
        return gene

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

    model.config.use_cache = use_cache 
    return float(count)/total_params 

tokenizer = AutoTokenizer.from_pretrained(FULL_MODEL_PATH)

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
evaluator = Evaluator(dataset, tokenizer, "cuda", n_samples=141)
models = load_sparse_models()

gene = load_last_gene(args.gene_path)
model = assemble_model(models, gene)

model.save_pretrained(args.output_folder + str(args.base_sparsity))
tokenizer.save_pretrained(args.output_folder + str(args.base_sparsity))

ppl = evaluator.evaluate(model)

print(ppl)

sparsity_ratio = check_sparsity(model)
assert sparsity_ratio == args.base_sparsity / 64