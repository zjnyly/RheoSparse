import math
import time

import torch
import torch.nn as nn
import transformers
import numpy as np
import os

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class SparseGPT:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune_(
        self, args, base_sparsity, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, name = "", layer = 0
        # self,sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, name = "", layer = 0
    ):
        print("pruning layer ", name, base_sparsity)
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None
        if 'gate' not in name and 'up' not in name and 'down' not in name:
            
            folder = args.model.split("/")[-1] + "_dp_path" + f"/{base_sparsity}"
            filename = f"layer_{layer}_{name}.npy"
            full_path = os.path.join(folder, filename)
            # searched_sparsity = np.load("Qwen-32-Searched-Sub-Sparsity/layer_" + str(layer) + "_" + name+ ".npy")
            searched_sparsity = np.load(full_path)
        
        N_TILES = W.shape[1] // 64  # Horizontal tiles
        M_TILES = W.shape[0] // 128  # Vertical tiles
        
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]


            mask1 = torch.zeros_like(W1) == 1
            
            # print(W1.shape)
            
            my_mask1 = torch.zeros_like(W1) == 1
            # for sub_block_row in range(0, self.rows // 128):
            #     for sub_block_col in range(0, blocksize // 64):
            #         global_block_row = sub_block_row
            #         global_block_col = (i1 // 64) + sub_block_col
            #         global_idx = global_block_row * N_TILES + global_block_col
            #         # sparsity_n = searched_sparsity[global_idx]
            #         sparsity_n = 32
            #         # if(sub_block_col == 1):
            #         #     print(((torch.diag(Hinv1)[sub_block_col * 64:(sub_block_col + 1) * 64].reshape((1, -1))) ** 2))
            #         #     exit()
            #         cur_tmp = (W1[sub_block_row * 128:(sub_block_row + 1) * 128, sub_block_col * 64:(sub_block_col + 1) * 64] ** 2) / ((torch.diag(Hinv1)[sub_block_col * 64:(sub_block_col + 1) * 64].reshape((1, -1))) ** 2)
                    
            #         my_mask1[sub_block_row * 128:(sub_block_row + 1) * 128].scatter_(1, sub_block_col * 64 + torch.topk(cur_tmp, sparsity_n, dim=1, largest=False)[1], True)
            
            # for sub_block_row in range(0, self.rows // 128):
            #     sub_block_col = (i // 64)
            #     global_block_row = sub_block_row
            #     global_block_col = (i1 // 64) + sub_block_col
            #     global_idx = global_block_row * N_TILES + global_block_col
            #     # sparsity_n = searched_sparsity[global_idx]
            #     sparsity_n = 32
            #     cur_tmp = (W1[sub_block_row * 128:(sub_block_row + 1) * 128, sub_block_col * 64:(sub_block_col + 1) * 64] ** 2) / ((torch.diag(Hinv1)[sub_block_col * 64:(sub_block_col + 1) * 64].reshape((1, -1))) ** 2)
            #     my_mask1[sub_block_row * 128:(sub_block_row + 1) * 128].scatter_(1, sub_block_col * 64 + torch.topk(cur_tmp, sparsity_n, dim=1, largest=False)[1], True)
                    
                    
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    
                    
                    for sub_block_row in range(0, self.rows // 128):
                        sub_block_col = (i // 64)
                        global_block_row = sub_block_row
                        global_block_col = (i1 // 64) + sub_block_col
                        global_idx = global_block_row * N_TILES + global_block_col
                        if 'gate' not in name and 'up' not in name and 'down' not in name:
                            sparsity_n = searched_sparsity[global_idx] + base_sparsity - 4 
                        else:
                        # print(global_idx)
                            sparsity_n = base_sparsity
                        cur_tmp = (W1[sub_block_row * 128:(sub_block_row + 1) * 128, sub_block_col * 64:(sub_block_col + 1) * 64] ** 2) / ((torch.diag(Hinv1)[sub_block_col * 64:(sub_block_col + 1) * 64].reshape((1, -1))) ** 2)
                        mask1[sub_block_row * 128:(sub_block_row + 1) * 128].scatter_(1, sub_block_col * 64 + torch.topk(cur_tmp, sparsity_n, dim=1, largest=False)[1], True)
                
                    # tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                   
                    # mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)
                    # status = torch.equal(my_mask1[:128, i:(i + prune_m)], mask1[:128, i:(i + prune_m)])
                    # difference_mask = (my_mask1[:128, i:(i + prune_m)] != mask1[:128, i:(i + prune_m)])
                    # num_differences = torch.sum(difference_mask).item()
                    # # print(num_differences)
                    # # assert status
                    

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])
        # exit()

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        loss = torch.sum(Losses).item()
        return loss
    
        # self, args, base_sparsity, sparsity, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, name = "", layer = 0
    
    def fasterprune(
        self, sparsity=0.5, prune_n=0, prune_m=0, blocksize=128, percdamp=.01, name = "", layer = 0
    ):
        
        
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        mask = None

        print("pruning layer ", name, " sparsity: ", sparsity, " prune_n: ", prune_n, " prune_m: ", prune_m, " self.columns: ", self.columns, " percdamp: ", percdamp)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prune_n == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prune_n != 0 and i % prune_m == 0:
                    tmp = W1[:, i:(i + prune_m)] ** 2 / (torch.diag(Hinv1)[i:(i + prune_m)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prune_n, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d 
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        print("sparsity", (W==0).sum().item() / W.numel())
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        loss = torch.sum(Losses).item()
        return loss

    def free(self):
        self.H = None
        torch.cuda.empty_cache()