import time 
import heapq 
import torch 
import torch.nn as nn 
import numpy as np
import json
from sparsegpt import SparseGPT 
from data import get_loaders 
import os

#from .ablate import AblateGPT 

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

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((512, 2048, model.config.hidden_size), dtype=dtype, device=device) # ori: 128
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
   
    for batch in dataloader:
        # print(batch[0].shape)
        try:
            model(batch[0].to(device))
        except ValueError:
            pass 
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids 

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity



@torch.no_grad()
def prune_sparsegpt_(args, model, tokenizer, dev, prune_n, prune_m, base_sparsity, sparsity):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    args.nsamples = 512
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    # dataloader, _ = get_loaders("ultra",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')
    #  15.219374656677246 original pruned 32
    #  15.113567352294922 with sparsity result from wanda 
    # gene = [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -6, 0, 0, -1, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -1, 2, 0, 0, 0, -2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 0, 0, 0, -2, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 1, 4, 0, 2, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -2, 0, 0, 0, 2, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 0, 0, -4, 0, 0, 2, 1, 0, 0, -2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -2, 4, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, -2, 0, 0, 0, 0, -4, 0, 2, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
        subset = find_layers(layer)
        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        for name in gpts:
            # prune_n = 32
            # prune_m = 64
            # if 'q_proj' in name:
            #     prune_n = base_sparsity + gene[i]
            # elif 'k_proj' in name:
            #     prune_n = base_sparsity + gene[28 + i]
            # elif 'v_proj' in name:
            #     prune_n = base_sparsity + gene[28 * 2 + i]
            # elif 'o_proj' in name:
            #     prune_n = base_sparsity + gene[28 * 3 + i]
            # elif 'gate' in name:
            #     prune_n = base_sparsity + gene[28 * 4 + i]
            # elif 'up' in name:
            #     prune_n = base_sparsity + gene[28 * 5 + i]
            # elif 'down' in name:
            #     prune_n = base_sparsity + gene[28 * 6 + i]
            loss = gpts[name].fasterprune(args, base_sparsity, sparsity, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128, name = name, layer = i)
            # if "gate" in name: print(loss)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
    

@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n, prune_m):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    args.nsamples = 512
    
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, 2048, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask = inps.to(dev), outs.to(dev), attention_mask.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            loss = gpts[name].fasterprune(prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128, name = name, layer = i)
            
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]
        self.columns = layer.weight.data.shape[1]

        self.scaler_row = torch.zeros((self.columns), device=self.dev)
        self.nsamples = 0

        self.layer_id = layer_id 
        self.layer_name = layer_name

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)
        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples 


@torch.no_grad()
def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), gene_sequence = 'PATH', base_sparsity = 32):
    layers = model.model.layers 
    num_layers = len(layers)

    gene = [0, 0, 0, 0, 2, 0, -1, -4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, -1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, -4, 0, 0, 0, 0, 2, 0, 0, -1, -2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -4]
    
    # wikitext perplexity 124.41925811767578
    # wikitext perplexity 126.34646606445312
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            prune_n = 32
            prune_m = 64
            # if 'gate' in name:
            #     prune_n = base_sparsity + gene[i]
            # elif 'up' in name:
            #     prune_n = base_sparsity + gene[num_layers + i]
            # elif 'down' in name:
            #     prune_n = base_sparsity + gene[num_layers * 2 + i]
                
            print(name, prune_n, prune_m)
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


def compute_sparsity_mask(W_tile, sparsity, dir):
    """Compute sparsity mask for a tile at a given sparsity level (e.g., 32 for 32%)."""
    W_mask = torch.zeros_like(W_tile) == 1  # Initialize mask (all False)
    num_cols = W_tile.shape[1]
    prune_n = sparsity  # Number of unmasked (kept) columns
    prune_m = num_cols  # Use full tile width as pruning block
    
    if prune_n != 0:
        for ii in range(0, num_cols, prune_m):
            tmp = W_tile[:, ii:(ii+prune_m)].float()
            indices = ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1]
            W_mask.scatter_(1, indices, True)
    
    if dir == 0:
        return ~W_mask
    else:
        return W_mask


def min_loss_with_sparsity_constraint(loss, base_sparsity, num_blocks, layer_weight):
    n = loss.shape[0]  # Number of layers
    m = loss.shape[1]  # Number of sparsity choices (9)

    # Define sparsity choices: base_sparsity ± 4 (28 to 36 for base=32)
    sparsity_choices = [base_sparsity + i for i in range(-4, 5)]  # e.g., [28, 29, 30, 31, 32, 33, 34, 35, 36]
    
    # Total sparsity constraint: base_sparsity * num_layers
    total_sparsity = base_sparsity * num_blocks

    # Costs represent sparsity values for each choice
    int_costs = np.zeros((n, m), dtype=int)
    for i in range(n):
        for j in range(m):
            int_costs[i][j] = sparsity_choices[j] * layer_weight

    # Integerize costs to avoid floating-point issues
    int_total_sparsity = int(total_sparsity)

    # # DP table: dp[i][s]: min loss for first i layers with sparsity sum s
    dp = np.full((n + 1, int_total_sparsity + 1), -float('inf'))
    prev = np.full((n + 1, int_total_sparsity + 1), -1, dtype=int)  # For backtracking

    # Initialize
    dp[0][0] = 0

    # print(n, " ",  int_total_sparsity, " ",  m)
    # DP
    for i in range(1, n + 1):
        # print(i , end=" ", flush=True)
        for s in range(int_total_sparsity + 1):
            for j in range(m):
                if s >= int_costs[i-1][j] and dp[i-1][s - int_costs[i-1][j]] != -float('inf'):
                    new_loss = dp[i-1][s - int_costs[i-1][j]] + loss[i-1][j]
                    if new_loss > dp[i][s]:
                        dp[i][s] = new_loss
                        prev[i][s] = j
    # print(" ")
    # Check if exact sparsity sum is achievable
    max_loss = dp[n][int_total_sparsity]
    if max_loss == -float('inf'):
        return None, None, None  # No solution

    # Backtrack to find path
    path = []
    s = int_total_sparsity
    for i in range(n, 0, -1):
        j = prev[i][s]
        path.append(j)
        s -= int_costs[i-1][j]

    path.reverse()
    return max_loss, path, sum(sparsity_choices[j] for j in path)

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, base_sparsity = 32):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    args.nsamples=512
    print("loading calibdation data")
    #dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    dataloader, _ = get_loaders("ultra",nsamples=args.nsamples,seed=args.seed,seqlen=2048,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)
        
    
    # gene = [0, 0, 0, 0, 2, 0, -1, -4, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, -1, 0, 0, 0, 3, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, -4, 0, 0, 0, 0, 2, 0, 0, -1, -2, 0, 0, 4, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -4]
    # prune_n = 32
    # prune_m = 64

    layers = model.model.layers
    num_layers = len(layers)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        # wikitext perplexity 15.24555778503418  global searched with gene improvement
        # wikitext perplexity 15.562688827514648 global searched
        # wikitext perplexity 15.610438346862793 unstr
        # wikitext perplexity 15.606783866882324 searched + dp
        # wikitext perplexity 15.878556251525879 searched 32:64
        # wikitext perplexity 15.896939277648926 dp 32:64
        # wikitext perplexity 16.183609008789062 vanilla 32:64
        for name in subset:
            prune_n = 0
            prune_m = 0
            # prune_n = 0
            
            # if 'gate' in name:
            #     prune_n = base_sparsity + gene[i]
            #     # args.sparsity_ratio = prune_n / prune_m
            #     # prune_n = 0
            # elif 'up' in name:
            #     prune_n = base_sparsity + gene[num_layers + i]
            #     # args.sparsity_ratio = prune_n / prune_m
            #     # prune_n = 0
            # elif 'down' in name:
            #     prune_n = base_sparsity + gene[num_layers * 2 + i]
            #     # args.sparsity_ratio = prune_n / prune_m
            #     # prune_n = 0
            # print(f"pruning layer {i} name {name}", prune_n, prune_m, args.sparsity_ratio)
            print(f"pruning layer {i} name {name}", base_sparsity)
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
            # if 'gate' not in name and 'up' not in name and 'down' not in name:
            if False:
                N_TILES = subset[name].weight.data.shape[1] // 64  # Horizontal tiles
                M_TILES = subset[name].weight.data.shape[0] // 128  # Vertical tiles
                NUM_BLOCKS = N_TILES * M_TILES
                BASE_SPARSITY = base_sparsity  # Base sparsity value
                SPARSITY_RANGE = range(-4, 5)  # Sparsity levels: 28 to 36
                num_tiles = N_TILES * M_TILES
                num_sparsity_levels = len(SPARSITY_RANGE)  # 9
                loss_data = np.zeros((num_tiles, num_sparsity_levels))
                
                TILE_HEIGHT = W_metric.shape[0] // M_TILES
                TILE_WIDTH = W_metric.shape[1] // N_TILES
                    
                
                
                tile_idx = 0
                for m in range(M_TILES):
                    for n in range(N_TILES):
                        # print(m * N_TILES + n, end = " ")
                        # Extract tile
                        tile_row_start = m * TILE_HEIGHT
                        tile_row_end = (m + 1) * TILE_HEIGHT
                        tile_col_start = n * TILE_WIDTH
                        tile_col_end = (n + 1) * TILE_WIDTH
                        W_tile = W_metric[tile_row_start:tile_row_end, tile_col_start:tile_col_end]
                        
                        # Compute masks and losses for each sparsity level
                        for s_idx, s_offset in enumerate(SPARSITY_RANGE):
                            sparsity = BASE_SPARSITY + s_offset  # e.g., 28, 36
                            W_mask = compute_sparsity_mask(W_tile, sparsity, 0)
                            # Loss = sum of W_metric values where mask is True (unmasked)
                            loss = W_tile[W_mask].sum().item()
                            loss_data[tile_idx][s_idx] = loss
                        tile_idx += 1
                # print("end of prepare")       
                min_loss, path, used_sparsity = min_loss_with_sparsity_constraint(loss_data, base_sparsity, NUM_BLOCKS, 1)
                # Qwen-32-Searched-Sub-Sparsity
                # np.save("Qwen-32-Searched-Sub-Sparsity/"+ str(base_sparsity) + "/" + "layer_" + str(i) + "_" + name, np.array(path)) 
                
                # if args.save_dp_path:
                folder = args.model.split("/")[-1] + "_dp_path" + f"/{base_sparsity}"
                filename = f"layer_{i}_{name}.npy"
                full_path = os.path.join(folder, filename)

                # 创建目录（如果不存在）
                os.makedirs(folder, exist_ok=True)

                # 保存文件
                np.save(full_path, np.array(path))
                tile_idx = 0
                for m in range(M_TILES):
                    for n in range(N_TILES):
                        # print(m * N_TILES + n, end = " ")
                        # Extract tile
                        tile_row_start = m * TILE_HEIGHT
                        tile_row_end = (m + 1) * TILE_HEIGHT
                        tile_col_start = n * TILE_WIDTH
                        tile_col_end = (n + 1) * TILE_WIDTH
                        W_tile = W_metric[tile_row_start:tile_row_end, tile_col_start:tile_col_end]
                        
                        sparsity = BASE_SPARSITY + path[tile_idx] - 4  # e.g., 28, 36
                        W_mask = compute_sparsity_mask(W_tile, sparsity, 1)
                        tile_idx += 1
                        W_tile[W_mask] = 0
                        subset[name].weight.data[tile_row_start:tile_row_end, tile_col_start:tile_col_end][W_mask] = 0
            else:
                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:,ii:(ii+prune_m)].float()
                            W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        print("use_variant")
                        # wanda variant 
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0., 0.8]
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                            if cur_sparsity > args.sparsity_ratio:
                                alpha_new = (alpha + alpha_hist[0]) / 2.0
                                alpha_hist[1] = alpha
                            else:
                                alpha_new = (alpha + alpha_hist[1]) / 2.0
                                alpha_hist[0] = alpha

                            alpha = alpha_new 
                            W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                    else:
                        # unstructured pruning
                        indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

                subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()


