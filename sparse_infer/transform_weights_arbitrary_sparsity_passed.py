import torch
import torch.nn as nn
import numpy as np
import os
from transformers import AutoModelForCausalLM
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--sparse_path', type=str)
parser.add_argument('--base_sparsity', type=int)
parser.add_argument('--num_layers', type=int)

args = parser.parse_args()



N = 32  
BLOCK_SIZE = 64  
GROUP_SIZE = 128  

def load_last_gene(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1].strip()  # 获取最后一行并去除换行符
        data = json.loads(last_line)  # 解析 JSON
        gene = data['gene']           # 提取 gene 数组
        return gene


def process_weight(weight_fp16, SPARSITY=32):
    assert weight_fp16.dim() == 2
    weight_np = weight_fp16.cpu().numpy()
    rows, cols = weight_np.shape
    num_blocks = cols // BLOCK_SIZE

    packed_weights = []
    packed_masks = []

    for row in weight_np:
        row_blocks = np.split(row, num_blocks)
        row_sparse_vals = []
        row_masks = []
        for block in row_blocks:
            mask = (block != 0).astype(np.uint64)
            # print(mask.sum(), SPARSITY)
            assert mask.sum() == 64-SPARSITY
            bits = 0
            for i, bit in enumerate(mask):
                bits |= (int(bit) << i)
            row_masks.append(bits)
            row_sparse_vals.extend(block[mask == 1])
        
        packed_weights.append(np.array(row_sparse_vals, dtype=np.float16))
        packed_masks.append(np.array(row_masks, dtype=np.uint64))

    return packed_weights, packed_masks

                
def save_layer_sparse(WEIGHT_FP16, HEIGHT=None, WIDTH=None, SPARSITY=32):
    packed_weights, packed_masks = process_weight(WEIGHT_FP16, SPARSITY)
    packed_weights_np = np.stack(packed_weights, axis=0)
    packed_masks_np = np.stack(packed_masks, axis=0)
    HEIGHT_WISE_BLOCKS = HEIGHT // GROUP_SIZE
    WIDTH_WISE_BLOCKS = WIDTH // BLOCK_SIZE
    SPARSE_BLOCK_SIZE = 64-SPARSITY
    
    packed_weights_np = packed_weights_np.reshape([HEIGHT_WISE_BLOCKS, GROUP_SIZE, WIDTH_WISE_BLOCKS, SPARSE_BLOCK_SIZE]).transpose([0, 2, 1, 3])
    # now [HEIGHT_WISE_BLOCKS, WIDTH_WISE_BLOCKS, GROUP_SIZE, SPARSE_BLOCK_SIZE]
    packed_masks_np = packed_masks_np.reshape([HEIGHT_WISE_BLOCKS, GROUP_SIZE, WIDTH_WISE_BLOCKS]).transpose([0, 2, 1])
    
    valid_slices = []
    ptr = 0
    if SPARSE_BLOCK_SIZE >> 5 & 1:
        packed_weights_32 = packed_weights_np[:,:,:,:32].reshape([HEIGHT_WISE_BLOCKS, WIDTH_WISE_BLOCKS, -1])
        valid_slices.append(packed_weights_32)
        ptr += 32
    if SPARSE_BLOCK_SIZE >> 4 & 1:
        packed_weights_16 = packed_weights_np[:,:,:,ptr:ptr+16].reshape([HEIGHT_WISE_BLOCKS, WIDTH_WISE_BLOCKS, -1])
        valid_slices.append(packed_weights_16)
        ptr += 16
    if SPARSE_BLOCK_SIZE >> 3 & 1:
        packed_weights_8 = packed_weights_np[:,:,:,ptr:ptr+8].reshape([HEIGHT_WISE_BLOCKS, WIDTH_WISE_BLOCKS, -1])
        valid_slices.append(packed_weights_8)
        ptr += 8
    if SPARSE_BLOCK_SIZE >> 2 & 1:
        packed_weights_4 = packed_weights_np[:,:,:,ptr:ptr+4].reshape([HEIGHT_WISE_BLOCKS, WIDTH_WISE_BLOCKS, -1])
        valid_slices.append(packed_weights_4)
        ptr += 4
    if SPARSE_BLOCK_SIZE >> 1 & 1:
        packed_weights_2 = packed_weights_np[:,:,:,ptr:ptr+2].reshape([HEIGHT_WISE_BLOCKS, WIDTH_WISE_BLOCKS, -1])
        valid_slices.append(packed_weights_2)
        ptr += 2
    if SPARSE_BLOCK_SIZE & 1:
        packed_weights_1 = packed_weights_np[:,:,:,ptr:ptr+1].reshape([HEIGHT_WISE_BLOCKS, WIDTH_WISE_BLOCKS, -1])
        valid_slices.append(packed_weights_1)
        ptr += 1
    
    concatenated = np.concatenate(valid_slices, axis=-1)
    
    return concatenated, packed_masks_np
    
    # with open(f"{NAME}_weights.bin", "wb") as f_w:
    #     f_w.write(packed_weights_np.tobytes()) 
    # with open(f"{NAME}_masks.bin", "wb") as f_m:
    #     f_m.write(packed_masks_np.tobytes()) 
    
    


# 遍历模型
def export_sparse_model(model, compressed_state_dict, sparse_path, base_sparsity, num_layers):
    
    sparse_path = np.array(sparse_path).reshape([-1, num_layers]).transpose().flatten()

    
    
    cnt = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            print(name, sparse_path[cnt])
            cur_sparsity = sparse_path[cnt] + base_sparsity
            print(cur_sparsity)
            cnt += 1
            weight = module.weight.data.to(torch.float16)
            bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
            packed_weights_np, packed_masks_np = save_layer_sparse(weight, HEIGHT=weight.shape[0], WIDTH=weight.shape[1], SPARSITY=cur_sparsity)
            compressed_state_dict[name + "_decode.packed_weights"] = packed_weights_np.flatten()
            compressed_state_dict[name + "_decode.packed_mask"] = packed_masks_np.flatten()
            compressed_state_dict[name + "_decode.bias"] = bias
            
      




sparse_path = load_last_gene(args.sparse_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=torch.float16,
    device_map="cuda:0",
)


compressed_state_dict = {}
export_sparse_model(model, compressed_state_dict, sparse_path, args.base_sparsity, args.num_layers)
np.save(f"sparse_weights_{args.base_sparsity}.npy", compressed_state_dict)

